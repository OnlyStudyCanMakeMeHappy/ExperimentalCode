import argparse
import time
import torch.nn.functional
import losses
from model.Model import MultiTaskModel
from utils.common import *
from utils.data_utils import *
from utils.MPerClassSampler import MPerClassSampler
from torch.utils.tensorboard import SummaryWriter
from datasets import FGNET, Adience
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
                    } , action = ChoiceAction)
parser.add_argument('--dim', default=64, type=int, help='embedding size')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-j', '--workers', default=2, type=int)
parser.add_argument('-M', default = 6, type = int , help = 'MPerClassSampler')

#==============================validate and save model==========================#
parser.add_argument('--val_epoch' , default = 3, type = int)
parser.add_argument('--save_path' , default = 'result')
#==============================batch size==========================#
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--eval_batch_size', default=120, type=int)
#==============================learning rate==========================#
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--lr_scheduler', '-ls', dest = 'ls', action = "store_true") #dest: specify the attribute name used in the result name
parser.add_argument('--lr_decay_epochs' , default = [30 , 60, 90] , nargs = "+", help = "numbers list of learning rate decay epoch")
parser.add_argument('--lr_decay_gamma' , default = 0.1 , type=float)

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


def train(
        model,
        loss_funcA,
        loss_funcB,
        rel_train_loader,
        abs_train_loader ,
        aug_transform,
        optimizer,
        epoch,
        writer,
        logger,
):
    start_time = time.time()
    model.train()
    freeze_BN(model)
    lossA = AverageMeter()
    lossR = AverageMeter()
    gradAccum_R , gradAccum_A = len(rel_train_loader) , len(abs_train_loader)
    # 梯度清零
    optimizer.zero_grad()

    if args.fuse:
        for idx, (data , label) in enumerate(rel_train_loader):
            data = data.to(args.gpu , non_blocking = True)
            pairs_indices , pairs_labels =  construct_partial_pairs(label)
            idx , idy = pairs_indices[ : , 0], pairs_indices[ : , 1]
            pairs_labels = pairs_labels.to(args.gpu)
            embedding = model(data , 1)
            x_embedding, y_embedding = embedding[idx] , embedding[idy]
            loss1 = loss_funcA(torch.cat((x_embedding, y_embedding), dim = 1), pairs_labels)
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
            # loss是平均损失, 当前mini-batch的损失 : batch_size * loss
            lossR.update(loss.item() / args.Lambda , label.size(0))
            loss = loss / gradAccum_R
            loss.backward()

    for idx,(image , label) in enumerate(abs_train_loader):
        image = image.to(args.gpu, non_blocking = True)
        label = label.to(args.gpu, non_blocking = True)
        embedding = model(image)
        loss = loss_funcB(embedding, label)
        lossA.update(loss.item() , label.size(0))
        loss = loss / gradAccum_A
        loss.backward()

    end_time = time.time()
    # 梯度更新
    optimizer.step()

    lossA , lossR = lossA.avg , lossR.avg
    if args.fuse:
        loss_total = lossA + args.Lambda * lossR
        writer.add_scalars("loss" , {
            "lossA" : lossA,
            "lossR": lossR,
            "loss_total": loss_total,
        }, epoch)
        logger.info(f" Epoch:{epoch} ==> lossA : {lossA} , lossR : {lossR}, loss_total : {loss_total} , time_cost : {end_time - start_time : .2f}s")
    else:
        writer.add_scalar('only_absolute_information', lossA, epoch)
        logger.info(f" Epoch:{epoch} ==> lossA : {lossA} , time_cost = {end_time - start_time:.2f}s")


def main():
    fix_seed(0)
    assert args.gpu is not None, "GPU is necessary"

    model = MultiTaskModel(f_dim = args.dim , g_dim = args.dim , g_hidden_size=512).to(args.gpu)
    base_transform, aug_transform, test_transform = get_transforms()

    dataset = args.data()
    #abs_train_dataset, pop_dataset, test_dataset = process_dataset(dataset , 0.9 , 0.1, tx = base_transform, ty = test_transform)
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

    samplerA = MPerClassSampler(
        abs_train_dataset.labels,
        batch_size = args.batch_size,
        m = args.M,
        iter_per_epoch = len(abs_train_dataset) // args.batch_size
    )
    #P = 6
    P = len(dataset.classes)
    K = 8
    batch_size = P * K
    samplerR = MPerClassSampler(
        abs_train_dataset.labels,
        batch_size = batch_size,
        m = K,
        iter_per_epoch = len(rel_train_dataset) // args.batch_size
        #iter_per_epoch = len(abs_train_dataset) // batch_size
    )
    abs_train_loader = DataLoader(
        abs_train_dataset,
        batch_size = args.batch_size,
        pin_memory = True,
        sampler = samplerA,
        num_workers = args.workers
    )
    rel_train_loader = DataLoader(
        abs_train_dataset,
        batch_size = batch_size,
        pin_memory=True,
        sampler=samplerR,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = args.eval_batch_size,
    )

    params_group = [
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': model.f_head.parameters(), 'lr': args.lr * 10},
        {'params': model.g_head.parameters(), 'lr': args.lr * 10}
    ]
    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer , milestones=args.lr_decay_epochs , gamma = args.lr_decay_gamma)

    loss_funcA , loss_funcB = losses.Tripletloss(args)
    if args.fuse:
        hparams = runtime_env(args, **details)
    else:
        hparams = runtime_env(args)
    writer = SummaryWriter(os.path.join("runs" , dict2str(hparams)))
    print("#" * 30 +" START TRAING " + "#" * 30)
    best_acc = 0.0
    best_epoch = 0
    patience = 10
    for epoch in range(1 , args.epochs + 1):
        #train(Backbone, Rel_Net, Abs_Net, loss_funcA, loss_funcB, rel_train_loader, abs_train_loader , aug_transform, optimizer, epoch, writer , logger)
        train(model, loss_funcA, loss_funcB, rel_train_loader, abs_train_loader, aug_transform,
              optimizer, epoch, writer, logger)
        if args.ls:
            lr_scheduler.step()
        if epoch % args.val_epoch == 0 or epoch == args.epochs:
            res = Evaluation(abs_train_loader, abs_train_loader, model, args.gpu)
            acc = res['ACC']
            if acc > best_acc:
                best_acc = acc
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
    record(hparams, eval_result)




if __name__ == "__main__":
    main()
