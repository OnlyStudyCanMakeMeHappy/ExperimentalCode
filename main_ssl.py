import argparse
import os
import time, math
import numpy as np
import torch
import torch.nn.functional as F
import losses
import datetime
import json

from model.Model import MultiTaskModel
from utils.common import *
from utils.data_utils import *
from torch.utils.tensorboard import SummaryWriter
from datasets import FGNET, Adience, UTKFace
from itertools import cycle
# 启动命令 : tensorboard --logdir=/path/to/logs/ --port=xxxx
parser = argparse.ArgumentParser(description="Train Model")


class ChoiceAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        """
        :param parser: The ArgumentParser object which contains this action
        :param namespace: The Namespace object that will be returned by parse_args().
        :param value: command-line arguments
        :return: set attributes on the namespace based on dest and values
        """
        setattr(namespace, self.dest, self.choices[value])


# worker -> windows : 0
parser.add_argument('--gpu', default=None, type=int, required=True)
parser.add_argument('--data', default=FGNET,
                    choices={
                        "FGNET": FGNET,
                        "Adience": Adience,
                        'UTKFace': UTKFace,
                    }, action=ChoiceAction)
parser.add_argument('--dim', default=128, type=int, help='embedding size')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--backbone', default='resnet50', type=str, help='The backbone of model')

# ==============================Record==========================#
parser.add_argument('--record', action='store_true')
parser.add_argument('--save_path', default='result')
parser.add_argument('--log_dir', default='runs')

# ==============================Sampler==========================#
parser.add_argument('--M', default=8, type=int, help='MPerClassSampler of absolute information')
parser.add_argument('--P', default=8, type=int)
parser.add_argument('--K', default=8, type=int)
parser.add_argument('--iter_per_epoch', default=None, type=int)

# ==============================validate and save model==========================#
parser.add_argument('--val_epoch', default=1, type=int)
parser.add_argument('--knn', dest='k', default=10, type=int)

# ==============================batch size==========================#
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--eval_batch_size', default=128, type=int)

# ==============================optimizer and scheduler==========================#
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--lr_scheduler', '-ls', dest='ls', default='multi_step', type=str,
                    choices=['multi_step', 'cosine_anneal'])  # dest: specify the attribute name used in the result name
parser.add_argument('--milestones', default=[30, 60, 90], nargs="+", type=int)
parser.add_argument('--lr_decay_gamma', default=0.1, type=float)
parser.add_argument('--warm_up_epochs', default=10, type=int)

# ==============================hyper parameters of relative information==============================#
parser.add_argument('--fuse', action="store_true", help="whether fuse the relative information")
parser.add_argument('--aug', action='store_true', help='Whether to do data augmentation for the head samples')
parser.add_argument('--loss', default='triplet', choices=['ms', 'triplet', 'margin'])
parser.add_argument('--mu', default=0.8, type=float)
parser.add_argument('--Lambda', default=1.0, type=float)
# 绝对信息的margin
parser.add_argument('--delta', default=0.1, type=float)
parser.add_argument('--vartheta', '-vt', default=0.1, type=float)
parser.add_argument('--varepsilon', '-ve', default=0.1, type=float)

args = parser.parse_args()


def dml_train(
        model,
        loss_func,
        abs_train_loader,
        optimizer,
):
    lossA = AverageMeter()
    ce_criterion = nn.CrossEntropyLoss()
    # for (image , label) in tqdm.tqdm(abs_train_loader, desc =f"Epoch:{epoch}/{args.epochs}" ,colour='blue' , ncols = 100, ascii = True):
    for image, label in abs_train_loader:
        image = image.to(args.gpu, non_blocking=True)
        label = label.to(args.gpu, non_blocking=True)
        #embedding = model(image)
        outputs = model(image)
        #loss_abs = loss_funcA(embedding_labeled, label)
        loss = ce_criterion(outputs, label)
        #loss = loss_func(embedding, label)
        lossA.update(loss.item(), label.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return lossA.avg


def main_and_aux_task_train(
        epoch,
        model,
        loss_funcA,
        rel_train_loader,
        abs_train_loader,
        optimizer
):
    w = 1.0
    if epoch < 30:
        epoch = np.clip(epoch, 0.0, 30)
        phase = 1.0 - epoch / 30
        w = float(np.exp(-5.0 * phase * phase))

    ce_criterion = nn.CrossEntropyLoss()
    cons_criterion = lambda logit1, logit2 : F.mse_loss(F.softmax(logit1,1), F.softmax(logit2,1))
    lossA = AverageMeter()
    lossR = AverageMeter()
    task_mask = [0] * len(abs_train_loader) + [1] * len(rel_train_loader)
    random.shuffle(task_mask)
    iterA, iterR = iter(abs_train_loader), iter(rel_train_loader)
    # asill = True, 123456789#
    # for task_id in tqdm.tqdm(task_mask, desc =f"Epoch:{epoch}/{args.epochs}" ,colour='green' , ncols = 80, ascii = True):
    for task_id in task_mask:
        if task_id == 1:
            (data,aug_data), label = next(iterR)
            data = data.to(args.gpu, non_blocking=True)
            aug_data = aug_data.to(args.gpu, non_blocking=True)
            unlab_outputs = model(data)
            pi_outputs = model(aug_data)
            cons_loss = cons_criterion(unlab_outputs, pi_outputs) * w
            lossR.update(cons_loss.item(), data.size(0))

            # loss = sup_loss + args.Lambda * w * cons_loss
            optimizer.zero_grad()
            cons_loss.backward()
            optimizer.step()

        else:
            image, label = next(iterA)
            image = image.to(args.gpu, non_blocking=True)
            label = label.to(args.gpu, non_blocking=True)
            outputs = model(image)
            # loss_abs = loss_funcA(embedding_labeled, label)
            sup_loss = ce_criterion(outputs, label)
            lossA.update(sup_loss.item(), label.size(0))

            optimizer.zero_grad()
            sup_loss.backward()
            optimizer.step()

    lossA, lossR = lossA.avg, lossR.avg
    return lossA, lossR

'''
def train(
        model,
        loss_funcA,
        rel_train_loader,
        abs_train_loader,
        optimizer,
):
    lossA = AverageMeter()
    lossR = AverageMeter()
    for (image, label),  ((data, aug_data),_)in zip(cycle(abs_train_loader) , rel_train_loader):
        image = image.to(args.gpu, non_blocking=True)
        label = label.to(args.gpu, non_blocking=True)
        feature_labeled = model.backbone(image)
        embedding_labeled = model.f_head(feature_labeled)
        loss_abs = loss_funcA(embedding_labeled, label)
        lossA.update(loss_abs.item(), label.size(0))

        data = data.to(args.gpu, non_blocking=True)

        feature_unlabeled = model.backbone(data)
        feature_all = torch.cat([feature_labeled , feature_unlabeled])

        embedding_unlabeled = model.f_head(feature_unlabeled)
        embedding_all = torch.cat([embedding_labeled , embedding_unlabeled])

        sigma = 1.0
        sim_mat = torch.exp(-torch.cdist(embedding_all , embedding_all) ** 2 / (2 * sigma ** 2))
        #loss_s = 0.5 *  torch.sum(torch.cdist(feature_all , feature_all) ** 2 * sim_mat)
        loss_s = 0.5 * torch.sum(torch.sum((feature_all[:, None, :] - feature_all[None, :, :]) ** 2, dim = - 1) * sim_mat)

        aug_data = aug_data.to(args.gpu, non_blocking = True)
        aug_embedding = model(aug_data)
        loss_aug = torch.mean(F.relu(F.pairwise_distance(embedding_unlabeled, aug_embedding) - 0.1))

        lossR.update(loss_s.item() + loss_aug.item())

        loss = loss_abs + args.Lambda * loss_s + args.Lambda * loss_aug
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return lossA.avg, lossR.avg
'''
def train(
        epoch,
        model,
        loss_funcA,
        rel_train_loader,
        abs_train_loader,
        optimizer,
):
    lossA = AverageMeter()
    lossR = AverageMeter()
    # exp{−5 * (1 − T)^2}
    w = 1.0
    if epoch < 30:
        epoch = np.clip(epoch, 0.0, 30)
        phase = 1.0 - epoch / 30
        w = float(np.exp(-5.0 * phase * phase))

    ce_criterion = nn.CrossEntropyLoss()
    cons_criterion = lambda logit1, logit2 : F.mse_loss(F.softmax(logit1,1), F.softmax(logit2,1))
    for (image, label),  ((data, aug_data),_)in zip(cycle(abs_train_loader) , rel_train_loader):
    #for (image, label), ((data, aug_data), _) in zip(abs_train_loader, rel_train_loader):
    #for (image , label) in abs_train_loader:
        image = image.to(args.gpu, non_blocking=True)
        label = label.to(args.gpu, non_blocking=True)
        outputs = model(image)
        #loss_abs = loss_funcA(embedding_labeled, label)
        sup_loss = ce_criterion(outputs, label)
        lossA.update(sup_loss.item(), label.size(0))

        # optimizer.zero_grad()
        # sup_loss.backward()
        # optimizer.step()

    #for ((data,aug_data),_) in rel_train_loader:
        data = data.to(args.gpu, non_blocking=True)
        aug_data = aug_data.to(args.gpu, non_blocking=True)
        unlab_outputs = model(data)
        pi_outputs = model(aug_data)
        cons_loss = cons_criterion(unlab_outputs, pi_outputs) * w
        lossR.update(cons_loss.item(), data.size(0))

        loss = sup_loss + args.Lambda * w * cons_loss

        optimizer.zero_grad()
        #cons_loss.backward()
        loss.backward()
        optimizer.step()


    return lossA.avg, lossR.avg
def main():
    fix_seed(0)

    #model = MultiTaskModel(f_dim=args.dim, g_dim=2048, g_hidden_size=512, Backbone=args.backbone).to(args.gpu)
    model = MultiTaskModel(f_dim = 8, g_dim=2048, g_hidden_size=512, Backbone=args.backbone).to(args.gpu)

    base_transform, aug_transform, test_transform = get_transforms()

    # ===================================================================================================#
    #                                         Dataset Related                                            #
    # ===================================================================================================#
    dataset = args.data()
    dataset_name = dataset.__class__.__name__
    '''
    train : test : val = 8 : 1 : 1
    labeled : unlabeled = 1 : 9
    '''
    spilt_datasets = process_dataset(dataset, 0.8, 0.1, base_transform, test_transform, aug_transform = aug_transform)
    #spilt_datasets = process_dataset(dataset, 0.8, 0.1, base_transform, test_transform, aug_transform = None)
    details_info_print(spilt_datasets, dataset.classes)
    abs_train_dataset, rel_train_dataset, test_dataset, val_dataset = spilt_datasets.values()

    # abs_train_dataset, rel_train_dataset, test_dataset = process_dataset(dataset, 0.8, 0.1, tx=base_transform,ty=test_transform)
    # getlogger
    logger = get_logger()

    abs_train_loader, abs_eval_loader, rel_train_loader, test_loader, val_loader = loader_init(
        args,
        abs_train_dataset,
        rel_train_dataset,
        test_dataset,
        val_dataset
    )
    # =======================================loss function==============================================#
    loss_funcR, loss_funcA = losses.Tripletloss(args)
    # ===================================================================================================#
    #                                                                                                   #
    #                       Optimizer and learning rate scheduler Configuration                         #
    #                                                                                                   #
    # ===================================================================================================#
    params_group = [
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': model.f_head.parameters(), 'lr': args.lr * 10},
        {'params': model.g_head.parameters(), 'lr': args.lr * 10},
    ]

    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)
    if args.ls == 'multi_step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.lr_decay_gamma)
        if args.warm_up_epochs:
            warm_up_with_multistep_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs \
                else args.lr_decay_gamma ** len([m for m in args.milestones if m <= epoch])
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

    elif args.ls == 'cosine_anneal':
        # 余弦退火，不重启
        # lr = et_min + 0.5 * (initial_et - et_min) * (1 + cos(pi * epoch / T_max))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.001,
                                                                  last_epoch=-1)
        if args.warm_up_epochs:
            et_min = args.lr * 0.001
            warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs \
                else et_min + 0.5 * (args.lr - et_min) * (
                        math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    hparams = runtime_env(args)

    print("#" * 10 + '=' * 20 + " START TRAING " + "=" * 20 + '#' * 10)

    best_metric = np.Inf
    # 将开始时间作为文件名
    if args.record:
        # result/Dataset_name/EXPmonthday/H:M
        now = datetime.datetime.now()

        time_str = now.strftime("%m%d-%H:%M")
        # runs/{dataset_name}/{date_time}
        day_str, _ = time_str.split('-')
        writer = SummaryWriter(os.path.join(args.log_dir, dataset_name, "EXP" + day_str, "EXP" + time_str))
        result_root = os.path.join(args.save_path, dataset_name, "EXP" + day_str, "EXP" + time_str)
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        # 创建实验目录
        # os.makedirs(result_root) , 递归创建

        with open(os.path.join(result_root, 'params.json'), 'w') as f:
            args.data = dataset_name
            json.dump(vars(args), f, indent=4)

        model_save_path = os.path.join(result_root, 'best_model.pth')
    ####

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        model.train()
        freeze_BN(model)
        if args.fuse:
            #lossA , lossR = main_and_aux_task_train(epoch,model,loss_funcA,rel_train_loader,abs_train_loader,optimizer)
            lossA, lossR = train(epoch ,model, loss_funcA, rel_train_loader, abs_train_loader,optimizer)
            loss_total = lossA + args.Lambda * lossR
            if args.record:
                writer.add_scalars("loss", {
                    "lossA": lossA,
                    "lossR": lossR,
                    "loss_total": loss_total,
                }, epoch)
            end_time = time.perf_counter()
            logger.info(
                f" Epoch:{epoch} ==> lossA : {lossA} , lossR : {lossR}, loss_total : {loss_total} , time_cost : {end_time - start_time : .2f}s")
        else:
            lossA = dml_train(model, loss_funcA, abs_train_loader, optimizer)
            end_time = time.perf_counter()
            if args.record:
                writer.add_scalar('only_absolute_information', lossA, epoch)
            logger.info(f" Epoch:{epoch} ==> lossA : {lossA} , time_cost = {end_time - start_time:.2f}s")

        if args.ls is not None:
            lr_scheduler.step()
        #            print(lr_scheduler.get_last_lr())
        if epoch % args.val_epoch == 0 or epoch == args.epochs:
            res = Evaluation(val_loader, abs_eval_loader, model, args)
            print(dict2str(res, '='))
            if args.record:
                writer.add_scalars("Metrics", res, epoch // args.val_epoch)
            metric = res['MAE']
            if metric < best_metric:
                best_metric = metric
                if args.record:
                    torch.save(model.state_dict(), model_save_path)
                    with open(os.path.join(result_root, 'best_result.txt'), 'w') as f:
                        f.write(f'Epoch : {epoch}\n')
                        for k, v in res.items():
                            f.write(f'{k} : {v}\n')

            # if acc < best_acc and epoch - best_epoch >= patience:
            #     print(f"Early Stopping at Epoch:{epoch}")
            #     break

    ### show the best metric
    if args.record:
        with open(os.path.join(result_root, 'best_result.txt')) as f:
            eval_results = ''.join(f.readlines()).replace('\n', ',').replace(':', '=')
            # logger.info("ACC = {ACC} , MAE = {MAE} , MSE = {MSE} , QWK = {QWK}, C-index = {C_index}".format(**eval_results))
            logger.info(eval_results)

    # checkpoint = torch.load(model_save_path)
    # model.load_state_dict(checkpoint)
    eval_results = Evaluation(test_loader, abs_eval_loader, model, args)
    record(vars(args), eval_results)


if __name__ == "__main__":
    print(vars(args))
    main()
