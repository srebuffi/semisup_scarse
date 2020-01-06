from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os.path
import dataset
from tqdm import tqdm
import models
import utils


def train(epoch, model, device, dataloader, optimizer, exp_lr_scheduler, criterion, args):
    """ Train loop, predict rotations. """
    loss_record = utils.AverageMeter()
    acc_record = utils.AverageMeter()
    save_path = args.save_dir + '/'
    os.makedirs(save_path, exist_ok=True)
    exp_lr_scheduler.step()
    model.train()
    for batch_idx, (data, label) in enumerate(tqdm(dataloader(epoch))):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)

        # measure accuracy and record loss
        acc = utils.accuracy(output, label)
        acc_record.update(100 * acc[0].item() / data.size(0), data.size(0))
        loss_record.update(loss.item(), data.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))

    return loss_record


def test(model, device, dataloader, args):
    """ Test loop, print metrics """
    acc_record = utils.AverageMeter()
    model.eval()
    for batch_idx, (data, label) in enumerate(tqdm(dataloader())):
        data, label = data.to(device), label.to(device)
        output = model(data)

        # measure accuracy and record loss
        acc = utils.accuracy(output, label)
        acc_record.update(100 * acc[0].item() / data.size(0), data.size(0))

    print('Test Acc: {:.4f}'.format(acc_record.avg))
    return acc_record


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='RotNet')
    parser.add_argument('--autoaugment', action='store_true', default=False,
                        help='Use autoaugment policy, only for CIFAR10 (Default: False)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='Input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset name (default: CIFAR10)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='Number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--network', type=str, default='ResNet-18',
                        help='Network model (default: ResNet-18), choose between (ResNet-18, TempEns, RevNet-18)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./data/rotNet',
                        help='Directory to save models')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    dataset_train = dataset.GenericDataset(dataset_name=args.dataset, split='train', autoaugment=args.autoaugment)
    dataset_test = dataset.GenericDataset(dataset_name=args.dataset, split='test')

    dloader_train = dataset.DataLoader(
        rotnet=True,
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    dloader_test = dataset.DataLoader(
        rotnet=True,
        dataset=dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)

    model = models.load_net(args.network)
    model = model.to(device)

    # follow the same setting as RotNet paper
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(args.epochs + 1):
        loss_record = train(epoch, model, device, dloader_train, optimizer, exp_lr_scheduler, criterion, args)
        acc_record = test(model, device, dloader_test, args)

        is_best = acc_record.avg > best_acc
        best_loss = max(acc_record.avg, best_acc)
        utils.save_checkpoint(model.state_dict(), is_best, args.save_dir, checkpoint='rotNet_%s_%s_lr_checkpoint.pth' % (args.dataset, args.network.lower()), best_model='rotNet_%s_%s_lr_best.pth' % (args.dataset, args.network.lower()))

        # Saving milestones only
        if epoch in [59, 119, 159, 199]:
            print('Saving model at milestone: %u' % (epoch))
            utils.save_checkpoint(model.state_dict(), False, args.save_dir, checkpoint='rotNet_%s_%s_%u_checkpoint.pth' % (args.dataset, args.network.lower(), epoch))


if __name__ == '__main__':
    main()
