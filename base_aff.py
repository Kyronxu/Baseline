# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2024/1/12 18:19
# Author     ：XuJ1E
# version    ：python 3.8
# File       : base_aff.py
"""
import argparse
import time
import sys

sys.path.append('.')
import yaml
import torch.utils.data
from loguru import logger

from utils.utli import *
from utils.metric import ConfusionMatrix
from utils.loss import LabelSmoothingCrossEntropy, CenterLoss
from torchvision import transforms, datasets
from torchsampler import ImbalancedDatasetSampler
from models.convnext import convnext_base

parser = argparse.ArgumentParser(description='Multi-task for multi-label of FER')
parser.add_argument('--data_path', default='/home/xuujie_ygc/workspace/MM_FOR_ML/data/AffectNet/', help='path to dataset')
parser.add_argument('--eval_only', default=True, action='store_true')
parser.add_argument('--eval_ckpt', default="/home/xuujie_ygc/workspace/MM_FOR_ML/FaceExpression/Baseline/weight/model/SIMPLE_MTL_AFFECTNET2_model_best.pth",
                                           type=str, help='checkpoint model for eval method')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--model_path', default='./weight/model', help='path for model checkpoint')
parser.add_argument('--num_classes', default=7, type=int, help='num_classes for prediction')
parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--arch', default='SIMPLE_MTL_AFFECTNET', type=str, help='model type for train')
parser.add_argument('--ratio', default=1, type=float, help='mask ratio for BEC loss function')
parser.add_argument('--drop_path', default=0.25, type=float, help='drop layer of model')
parser.add_argument('--lr', default=2.5e-4, type=float, help='initial learning rate')
parser.add_argument('--lr_new', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--decay-steps', default=None, type=int, help='decay step for learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=0.001, type=float, help='weight decay (default: 1e-4)', )
parser.add_argument('--print-freq', default=100, type=int, help='print frequency (default: 10)')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--seed', default=100002, type=int)


def main(args):
    exp_path = os.path.join('experiment/', time.strftime("%m_%d_%H_%M_%S", time.localtime()))
    os.makedirs(exp_path, exist_ok=True)
    logger.add(os.path.join(exp_path, 'log.txt'))

    seedForExp(args)
    logger.info('Epochs:', args.epochs, 'seed:', args.seed, 'Bs:', args.batch_size)
    logger.info("Use GPU: {} for training".format(args.gpu))
    logger.info(args)

    best_acc = 0.0
    model = convnext_base(pretrained=True, num_classes=args.num_classes, drop_path_rate=args.drop_path)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'),
                                         transform=transforms.Compose([
                                             transforms.Resize((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomApply([
                                                 transforms.RandomRotation(10),
                                                 transforms.RandomAffine(10, scale=(0.8, 1), translate=(0.2, 0.2))
                                             ], p=0.5),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             transforms.RandomErasing(p=0.8, scale=(0.05, 0.15))]))
    test_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'test'),
                                        transform=transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()

    optimizer = torch.optim.AdamW(params=model.module.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    if args.eval_only:
        model.eval()
        logger.info(f'Load model from {args.eval_ckpt}')
        model.load_state_dict(torch.load(args.eval_ckpt)['state_dict'])
        eval_one_epoch(test_loader, model)
        exit()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        t_acc, t_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
        e_acc, e_loss = eval_one_epoch(test_loader, model)
        scheduler.step()

        is_best = e_acc > best_acc
        best_acc = max(e_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }, is_best, args)
        #print(f">>>>>> >>>>>> best acc [{best_acc}]")

        logger.info(best_acc)
    with open(os.path.join(exp_path, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(vars(args), f)
    os.rename(exp_path, exp_path + '_%.2f' % best_acc)
    return best_acc


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Train acc', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()
        _, pred = model(images)
        acc, _ = accuracy(pred, target, topk=(1, 1))
        loss = criterion(pred, target)
        top1.update(acc[0], images.size(0))
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time()

        if i % args.print_freq == 0 and i != 0:
            progress.display(i)

    return top1.avg, losses.avg


def eval_one_epoch(test_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Train acc', ': 6.3f')
    progress = ProgressMeter(len(test_loader), [losses, top1], prefix="Validating")

    model.eval()
    confusion = ConfusionMatrix(num_classes=7, labels=['AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU'])
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda()
            target = target.cuda()
            _, pred = model(images)
            loss = criterion(pred, target)
            acc, _ = accuracy(pred, target, topk=(1, 1))
            top1.update(acc[0], images.size(0))
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            
            #if i % args.print_freq == 0 and i != 0:
            #    progress.display(i)
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            confusion.update(pred.cpu().numpy(), target.cpu().numpy())
        confusion.plot()
        confusion.summary()

        msg = 'Validating: Test_acc@1 {top1.avg:.3f} | Loss {losses.avg:.3f} | Batch_time {batch_time.sum:.1f}'.format(
            top1=top1, losses=losses, batch_time=batch_time)
        logger.info(msg)

        return top1.avg, losses.avg


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)