import argparse
import time
import torch.nn.functional

import losses
from model.embedding import Embedding
from model.mlp import MLP
import model.backbone as backbone
from utils.common import *
from utils.data_utils import *
from MPerClassSampler import MPerClassSampler
from torch.utils.tensorboard import SummaryWriter
from datasets import FGNET, Adience

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
        Backbone,
        Rel_Net,
        Abs_Net,
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
    Backbone.train()
    freeze_BN(Backbone)
    lossAList, lossRList = [] , []
    if args.fuse:
        for idx, (data, pairs_indices, label) in enumerate(rel_train_loader):
            data = data.to(args.gpu , non_blocking = True)
            idx , idy = pairs_indices[ : , 0], pairs_indices[ : , 1]
            label = label.to(args.gpu)
            embedding = Rel_Net(data)
            x_embedding, y_embedding = embedding[idx] , embedding[idy]
            loss1 = loss_funcA(torch.cat((x_embedding, y_embedding), dim = 1), label)
            aug_data = aug_transform(data)
            aug_embedding = Rel_Net(aug_data)[idx]
            loss2 = torch.mean(
                torch.nn.functional.relu(
                    torch.nn.functional.pairwise_distance(x_embedding, aug_embedding) -
                    torch.nn.functional.pairwise_distance(y_embedding, aug_embedding) +
                    args.varepsilon
                )
            )
            loss = loss1 + args.mu * loss2
            #loss = loss1
            lossRList.append(loss.item())
            loss *= args.Lambda
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for idx,(image , label) in enumerate(abs_train_loader):
        image = image.to(args.gpu, non_blocking = True)
        label = label.to(args.gpu, non_blocking = True)
        embedding = Abs_Net(image)
        loss = loss_funcB(embedding, label)
        lossAList.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    lossA , lossR = np.mean(lossAList) , np.mean(lossRList)
    loss_total = lossA + args.Lambda * lossR

    if args.fuse:
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
    Backbone = backbone.ResNet50().to(args.gpu)
    Abs_Head = Embedding(Backbone.output_size, embedding_size=args.dim).to(args.gpu)
    Rel_Head = MLP(Backbone.output_size, embedding_size=args.dim, hidden_size=512).to(args.gpu)
    Abs_Net = nn.Sequential(Backbone, Abs_Head)
    Rel_Net = nn.Sequential(Backbone, Rel_Head)
    #architecture.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))

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
            "The size of absolute information : {abs_info}"
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
        collate_fn = match_partial_pairs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = args.eval_batch_size,
    )

    params_group = [
        {'params': Backbone.parameters(), 'lr': args.lr},
        {'params': Abs_Head.parameters(), 'lr': args.lr * 10},
        {'params': Rel_Head.parameters(), 'lr': args.lr * 10}
    ]

    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer , milestones=args.lr_decay_epochs , gamma = args.lr_decay_gamma)

    loss_funcA , loss_funcB = losses.Tripletloss(args)
    if args.fuse:
        hparams = runtime_env(args, **details)
    else:
        hparams = runtime_env(args)
    writer = SummaryWriter(os.path.join("runs" , dict2str(hparams)))
    print("=" * 30 +" START TRAING" + "=" * 30)
    best_acc = 0.0
    best_epoch = 0
    patience = 10
    for epoch in range(1 , args.epochs + 1):
        train(Backbone, Rel_Net, Abs_Net, loss_funcA, loss_funcB, rel_train_loader, abs_train_loader , aug_transform, optimizer, epoch, writer , logger)
        if args.ls:
            lr_scheduler.step()
        if epoch % args.val_epoch == 0 or epoch == args.epochs:
            res = Evaluation(abs_train_loader, abs_train_loader, Abs_Net, args.gpu)
            acc = res['ACC']
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(Abs_Net.state_dict(), os.path.join(args.save_path , 'best_model.pth'))
                with open(os.path.join(args.save_path , 'best_result.txt'), 'w') as f:
                    f.write(f'best_epoch : {epoch}\n')
                    for k, v in res.items():
                        f.write(f'best {k} : {v}\n')
            if acc < best_acc and epoch - best_epoch >= patience:
                print(f"Early Stopping at Epoch:{epoch}")
                break
    checkpoint = torch.load(os.path.join(args.save_path, 'best_model.pth'))
    Abs_Net.load_state_dict(checkpoint)
    eval_result = Evaluation(test_loader, abs_train_loader, model = Abs_Net, device = args.gpu)
    logger.info("ACC = {ACC} , MAE = {MAE} , MSE = {MSE} , QWK = {QWK}, C-index = {C_index}".format(**eval_result))
    writer.add_hparams(hparams , eval_result)
    record(hparams, eval_result)




if __name__ == "__main__":
    main()
