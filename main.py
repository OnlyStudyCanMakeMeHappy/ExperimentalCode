import argparse
import time
import torch.nn.functional
import losses
from model.Model import MultiTaskModel
from utils.common import *
from utils.data_utils import *
from MPerClassSampler import MPerClassSampler
from torch.utils.tensorboard import SummaryWriter
from datasets import FGNET, Adience, UTKFace
from torch.utils.data import DataLoader

# 启动命令 : tensorboard --logdir=/path/to/logs/ --port=xxxx
parser = argparse.ArgumentParser(description="Train Model")
class ChoiceAction(argparse.Action):
    def __call__(self, parser , namespace, value, option_string = None):
        """
        :param parser: The ArgumentParser object which contains this action
        :param namespace: The Namespace object that will be returned by parse_args().
        :param value: command-line arguments
        :return: set attributes on the namespace based on dest and values
        """
        setattr(namespace, self.dest, self.choices[value])
# worker -> windows : 0
parser.add_argument('--data', default = FGNET,
                    choices = {
                        "FGNET" : FGNET,
                        "Adience" : Adience,
                        'UTKFace' : UTKFace,
                    } , action = ChoiceAction)
parser.add_argument('--dim', default=64, type=int, help='embedding size')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-j', '--workers', default=4, type=int)

#==============================Sampler==========================#
parser.add_argument('--M', default = 8, type = int , help = 'MPerClassSampler of absolute information')
parser.add_argument('--P', default = None, type = int)
parser.add_argument('--K', default = 8, type = int)
parser.add_argument('--iter_per_epoch', default=None, type=int)

#==============================validate and save model==========================#
parser.add_argument('--val_epoch' , default = 3, type = int)
parser.add_argument('--save_path' , default = 'result')

#==============================batch size==========================#
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--eval_batch_size', default=120, type=int)

#==============================learning rate==========================#
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--lr_scheduler', '-ls', dest = 'ls', default = None ,choices = ['multi_step' , 'cosine_anneal']) #dest: specify the attribute name used in the result name
parser.add_argument('--milestones' , default = [30 , 60, 90] , nargs = "+")
parser.add_argument('--lr_decay_gamma' , default = 0.1 , type=float)
parser.add_argument('--warm_up_epochs' , default = 10 , type=float)

parser.add_argument('--gpu', default=None, type=int, required = True)
#==============================hyper parameters of relative information==============================#
parser.add_argument('--fuse' , action = "store_true", help = "whether fuse the relative information")
parser.add_argument('--loss', default='triplet', choices=['ms', 'triplet', 'margin'])
parser.add_argument('--mu', default = 0.8,type = float)
parser.add_argument('--Lambda', default = 0.8,type = float)
# 绝对信息的margin
parser.add_argument('--delta', default= 0.5,type = float)
parser.add_argument('--vartheta', '-vt', default = 0.5,type = float)
parser.add_argument('--varepsilon', '-ve', default=0.5, type = float)

args = parser.parse_args()

def dml_train(
        model ,
        loss_func,
        abs_train_loader,
        optimizer,
        epoch
):
    lossA = AverageMeter()
    #for (image , label) in tqdm.tqdm(abs_train_loader, desc =f"Epoch:{epoch}/{args.epochs}" ,colour='blue' , ncols = 100, ascii = True):
    for image , label in abs_train_loader:
        image = image.to(args.gpu, non_blocking = True)
        label = label.to(args.gpu, non_blocking = True)
        embedding = model(image)
        loss = loss_func(embedding, label)
        lossA.update(loss.item() , label.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return lossA.avg

def main_and_aux_task_train(
        model,
        loss_funcR,
        loss_funcA,
        rel_train_loader,
        abs_train_loader ,
        aug_transform,
        optimizer,
        epoch
):
    lossA = AverageMeter()
    lossR = AverageMeter()
    task_mask = [0] * len(abs_train_loader) + [1] * len(rel_train_loader)
    random.shuffle(task_mask)
    iterA , iterR = iter(abs_train_loader) , iter(rel_train_loader)
    # asill = True, 123456789#
    #for task_id in tqdm.tqdm(task_mask, desc =f"Epoch:{epoch}/{args.epochs}" ,colour='green' , ncols = 80, ascii = True):
    for task_id in task_mask:
        if task_id == 1:
            data , label = next(iterR)
            data = data.to(args.gpu , non_blocking = True)
            # 在线构建偏序对
            pairs_indices , pairs_labels =  construct_partial_pairs(label , args.gpu)
            idx , idy = pairs_indices[ : , 0], pairs_indices[ : , 1]
            #pairs_labels = pairs_labels.to(args.gpu)
            embedding = model(data , task_id)
            x_embedding, y_embedding = embedding[idx] , embedding[idy]
            loss1 = loss_funcR(torch.cat((x_embedding, y_embedding), dim = 1), pairs_labels)
            aug_data = aug_transform(data)
            aug_embedding = model(aug_data , task_id)[idx]
            loss2 = torch.mean(
                torch.nn.functional.relu(
                    torch.nn.functional.pairwise_distance(x_embedding, aug_embedding) -
                    torch.nn.functional.pairwise_distance(y_embedding, aug_embedding) +
                    args.varepsilon
                )
            )
            loss = (loss1 + args.mu * loss2) * args.Lambda
            lossR.update(loss.item() / args.Lambda , label.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            image , label = next(iterA)
            image = image.to(args.gpu, non_blocking = True)
            label = label.to(args.gpu, non_blocking = True)
            embedding = model(image, task_id)
            loss = loss_funcA(embedding, label)
            lossA.update(loss.item() , label.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    lossA , lossR = lossA.avg , lossR.avg
    return lossA, lossR

def train(
    model,
    loss_funcR,
    loss_funcA,
    rel_train_loader,
    abs_train_loader ,
    aug_transform,
    optimizer,
    epoch,
):
    lossA = AverageMeter()
    lossR = AverageMeter()
    for (data, label) in rel_train_loader:
        data = data.to(args.gpu , non_blocking = True)
        pairs_indices, pairs_labels = construct_partial_pairs(label , args.gpu)
        idx , idy = pairs_indices[ : , 0], pairs_indices[ : , 1]
        #pairs_labels = pairs_labels.to(args.gpu)
        embedding = model(data , 1)
        x_embedding, y_embedding = embedding[idx] , embedding[idy]
        loss1 = loss_funcR(torch.cat((x_embedding, y_embedding), dim = 1), pairs_labels)
        aug_data = aug_transform(data)
        aug_embedding = model(aug_data , 1)[idx]

        loss2 = torch.mean(
            torch.nn.functional.relu(
                torch.nn.functional.pairwise_distance(x_embedding, aug_embedding) -
                torch.nn.functional.pairwise_distance(y_embedding, aug_embedding) +
                args.varepsilon
            )
        )
        loss = (loss1 + args.mu * loss2) * args.Lambda
        lossR.update(loss.item() / args.Lambda , label.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for (image , label) in abs_train_loader:
        image = image.to(args.gpu, non_blocking = True)
        label = label.to(args.gpu, non_blocking = True)
        embedding = model(image)
        loss = loss_funcA(embedding, label)
        lossA.update(loss.item() , label.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return lossA.avg , lossR.avg

def same_iterations_train(
        model,
        loss_funcR,
        loss_funcA,
        rel_train_loader,
        abs_train_loader ,
        aug_transform,
        optimizer,
        epoch
):
    lossA = AverageMeter()
    lossR = AverageMeter()

    for (data,labelR) ,(image , labelA) in zip(rel_train_loader, abs_train_loader):

            data = data.to(args.gpu , non_blocking = True)
            # 在线构建偏序对
            pairs_indices , pairs_labels =  construct_partial_pairs(labelR , args.gpu)
            idx , idy = pairs_indices[ : , 0], pairs_indices[ : , 1]
            embedding = model(data , 1)
            x_embedding, y_embedding = embedding[idx] , embedding[idy]
            loss1 = loss_funcR(torch.cat((x_embedding, y_embedding), dim = 1), pairs_labels)
            aug_data = aug_transform(data)
            aug_embedding = model(aug_data , 1)[idx]
            loss2 = torch.mean(
                torch.nn.functional.relu(
                    torch.nn.functional.pairwise_distance(x_embedding, aug_embedding) -
                    torch.nn.functional.pairwise_distance(y_embedding, aug_embedding) +
                    args.varepsilon
                )
            )

            lossR.update(loss1.item() + args.mu * loss2.item() , labelR.size(0))

            image = image.to(args.gpu, non_blocking = True)
            labelA = labelA.to(args.gpu, non_blocking = True)
            embedding = model(image)
            loss3 = loss_funcA(embedding, labelA)
            lossA.update(loss3.item() , labelA.size(0))
            optimizer.zero_grad()
            # 联合损失函数
            loss = (loss1 + args.mu * loss2) * args.Lambda + loss3
            loss.backward()
            optimizer.step()

    lossA , lossR = lossA.avg , lossR.avg
    return lossA, lossR
def main():
    fix_seed(0)

    model = MultiTaskModel(f_dim = args.dim , g_dim = args.dim , g_hidden_size=512).to(args.gpu)

    base_transform, aug_transform, test_transform = get_transforms()

    dataset = args.data()
    abs_train_dataset, rel_train_dataset, test_dataset = process_dataset(dataset, 0.9, 0.1, tx=base_transform,ty=test_transform)
    # getlogger
    logger = get_logger()
    details = None
    if args.fuse:
        details = {
            "dataset": len(dataset),
            "test": len(test_dataset),
            "abs_info": len(abs_train_dataset),
            "rel_info" : len(rel_train_dataset),
        }
        logger.critical(
            "The length of dataset : {dataset} , The size of test : {test} , "
            "The size of absolute information : {abs_info}, "
            "The size of relative information : {rel_info}".format(**details)
        )
######################################## Absolute information DataLoader #################################
    samplerA = MPerClassSampler(
        abs_train_dataset.labels,
        batch_size = args.batch_size,
        m = args.M,
        iter_per_epoch = len(abs_train_dataset) // args.batch_size
    )
    abs_train_loader = DataLoader(
        abs_train_dataset,
        batch_sampler=samplerA,
        pin_memory = True,
        num_workers = args.workers
    )

######################################## Relative information DataLoader #################################
    P = len(dataset.classes) if args.P is None else args.P
    batch_size = P * args.K
    iter_per_epoch = len(rel_train_dataset) // batch_size if args.iter_per_epoch is None else args.iter_per_eopch
    samplerR = MPerClassSampler(
        rel_train_dataset.labels,
        batch_size=batch_size,
        m = args.K,
        iter_per_epoch = iter_per_epoch
    )

    rel_train_loader = DataLoader(
        rel_train_dataset,
        pin_memory=True,
        batch_sampler=samplerR,
        num_workers=args.workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = args.eval_batch_size,
    )
######################################## Optimizer Configuration #################################
    params_group = [
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': model.f_head.parameters(), 'lr': args.lr * 10},
        {'params': model.g_head.parameters(), 'lr': args.lr * 10}
    ]

    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)
    if args.ls == 'multi_step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer , args.milestones, gamma=args.lr_decay_gamma)
    elif args.ls == 'cosine_anneal':
        # 余弦退火，不重启
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer , T_max = args.epochs, eta_min=0, last_epoch=-1)


    loss_funcR , loss_funcA = losses.Tripletloss(args)
    if args.fuse:
        hparams = runtime_env(args, **details)
    else:
        hparams = runtime_env(args)
    writer = SummaryWriter(os.path.join("runs" , dict2str(hparams)))
    print("#" * 30 +" START TRAING " + "#" * 30)

    best_metric = np.Inf
    best_epoch = 0
    patience = 10
    for epoch in range(1 , args.epochs + 1):
        start_time = time.time()
        model.train()
        freeze_BN(model)
        if args.fuse:
            #lossA , lossR = main_and_aux_task_train(model,loss_funcR,loss_funcA,rel_train_loader,abs_train_loader,aug_transform,optimizer,epoch)
            #lossA, lossR = same_iterations_train(model, loss_funcR, loss_funcA, rel_train_loader, abs_train_loader,aug_transform, optimizer, epoch)
            lossA , lossR = train(model,loss_funcR,loss_funcA,rel_train_loader,abs_train_loader,aug_transform,optimizer,epoch)
            loss_total = lossA + args.Lambda * lossR
            writer.add_scalars("loss", {
                "lossA": lossA,
                "lossR": lossR,
                "loss_total": loss_total,
            }, epoch)
            end_time = time.time()
            logger.info(
                f" Epoch:{epoch} ==> lossA : {lossA} , lossR : {lossR}, loss_total : {loss_total} , time_cost : {end_time - start_time : .2f}s")
        else:
            lossA = dml_train(model , loss_funcA, abs_train_loader, optimizer, epoch)
            end_time = time.time()
            writer.add_scalar('only_absolute_information', lossA, epoch)
            logger.info(f" Epoch:{epoch} ==> lossA : {lossA} , time_cost = {end_time - start_time:.2f}s")

        if args.ls is not None:
            lr_scheduler.step()
        if epoch % args.val_epoch == 0 or epoch == args.epochs:
            res = Evaluation(test_loader = test_loader, train_loader=abs_train_loader, model = model, device=args.gpu, k = 5)
            for k , v in res.items():
                print(f"{k} = {v}", end = ",")
            print()
            metric = res['MAE']
            #if acc > best_acc:
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.save_path , 'best_model.pth'))
                with open(os.path.join(args.save_path , 'best_result.txt'), 'w') as f:
                    f.write(f'best_epoch : {epoch}\n')
                    for k, v in res.items():
                        f.write(f'best {k} : {v}\n')
            # if acc < best_acc and epoch - best_epoch >= patience:
            #     print(f"Early Stopping at Epoch:{epoch}")
            #     break
    checkpoint = torch.load(os.path.join(args.save_path, 'best_model.pth'))
    model.load_state_dict(checkpoint)
    eval_result = Evaluation(test_loader, abs_train_loader, model, device = args.gpu)
    logger.info("ACC = {ACC} , MAE = {MAE} , MSE = {MSE} , QWK = {QWK}, C-index = {C_index}".format(**eval_result))
    writer.add_hparams(hparams , eval_result)
    #record(hparams, eval_result)
    record(vars(args), eval_result)




if __name__ == "__main__":
    print(vars(args))
    main()
