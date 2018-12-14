from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import nin
from models.nin import BinConv2d
from torch.autograd import Variable
sys.path.append('../..')
from sketchlib.sketch_coding_bf import sketch_transform

def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/nin.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # process the weights including binarization
        bin_op.binarization()
        
        # forwarding
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data[0],
                optimizer.param_groups[0]['lr']))
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
                                    
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * correct / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return

def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

def fix_layer(model, index):
    layer_list = get_layer_list(model, bin_op)
    for param in layer_list[index].parameters():
        param.requires_grad = False


def sketch_layer(model, index, com_ratio=0.7):
    #bin_op.binarization()
    layer_list = get_layer_list(model, bin_op)
    wei = layer_list[index].conv.weight

    #float_step = (float(wei.max()) - float(wei.min())) / (2.** 8 - 1.)
    
    #quan_wei = quantize(wei, num_bits=8,
    #                       min_value=float(wei.min()),
    #                       max_value=float(wei.max()))
    #print(wei.data.cpu().unique())
    #assert len(wei.data.cpu().unique()) == 2, "Weights is not qunatized to 1-bit!"
    #float_step = abs(wei.data.cpu().unique()[-1].item())
    #print(float_step)
    [a,b,c,d] = wei.shape
  
    tmp = wei.data.cpu().numpy().reshape((a,b*c*d))
    tmp = tmp / tmp[:,0:1] 
    
    #print(quan_wei.data.cpu().numpy().reshape((a,b*c*d)).shape)
    #print(type(quan_wei.data.cpu().numpy().reshape((a,b*c*d))))
    #np.save('to_yang',quan_wei.data.cpu().numpy().reshape((a,b*c*d)))
    wei.data=torch.Tensor(sketch_transform(tmp, com_ratio).reshape((a,b,c,d)))
    wei.cuda()

def get_layer_list(model, bin_op):
    bin_op.binarization()
    layer_list = []
    for m in model.modules():
        if isinstance(m, BinConv2d):
            layer_list += [m]
    return layer_list


if __name__=='__main__':
    
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='/home/simonx/Documents/data_xnor',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.005',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', 
             default='./nin.best.pth.tar',
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    args = parser.parse_args()
    print('==> Options:',args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    if not os.path.isfile(args.data+'/train_data'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')

    trainset = data.dataset(root=args.data, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
            shuffle=True, num_workers=2)

    testset = data.dataset(root=args.data, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'nin':
        model = nin.Net()
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

        optimizer = optim.Adam(params, lr=0.10,weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    #if args.evaluate:
    test()
    #exit(0)

    for i in range(len(get_layer_list(model,bin_op))):
        #print(get_layer_list(model,bin_op))
        bin_op.binarization()
        sketch_layer(model, i, 0.7)
        fix_layer(model, i)
        model.cuda()
        print(str(i)+'-Sketch: ')
        test()
        print('Start Retraining ...')
        for epoch in range(1, 100):
            train(epoch)
            test()
    
