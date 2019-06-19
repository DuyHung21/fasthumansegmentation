import os
from os.path import dirname, abspath
import time
import argparse

import cv2

import torch
import torch.nn as nn
import torch.optim as optim

from models.espmatting import SegMatingNet
from models.final2 import CustomNet
from utils.iou import iouEval
from utils.criteria import seg_matting_loss, TrainMode
from utils import VisualizeGraph as viz
from utils import data as MyDataLoader
from utils.custom_transforms import CustomTransform

def train_val_split(dataset, val_size):
    import random

    indexes = [i for i in range(len(dataset))]
    random.shuffle(indexes)
    num_train = int (len(indexes) * (1 - val_size))
    train_files = []
    val_files = []
    for i in range(len(indexes)):
        if i < num_train:
            train_files.append(dataset[indexes[i]])
        else:
            val_files.append(dataset[indexes[i]])

    return train_files, val_files

def train(args, train_loader, model, criterion, optimizer, epoch, mode):
    model.train()

    iouEvalTrain = iouEval(args['classes'])

    epoch_loss = []

    total_batches = len(train_loader)
    for i, (input_var, gt_seg, gt_alpha) in enumerate(train_loader):
        start_time = time.time()

        if args['onGPU']:
            input_var = input_var.cuda()
            gt_seg = gt_seg.cuda()
            gt_alpha = gt_alpha.cuda()

        # Run the model
        pred_seg, pred_alpha = model(input_var)

        # Set the grad to zero
        optimizer.zero_grad()
        loss = seg_matting_loss(input_var, pred_seg, gt_seg, pred_alpha, gt_alpha, mode)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        # Compute the confusion matrix
        iouEvalTrain.addBatch(pred_seg.max(1)[1].data, gt_seg[:, 0, :, :].data)

        print('[%d%d] loss: %3f time %.2f' % (i, total_batches, loss.item(), time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    '''
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    '''
    torch.save(state, filenameCheckpoint)

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def main(model_mode='esp', pretrained=None, seg=None, refine=None, all=None, start_seg=0, start_refine=0, start_all=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    width = 256
    height = 256
    batch_size = 32

    parrent_dir = dirname(dirname(abspath(__file__)))
    with open(os.path.join(parrent_dir, 'SegmentationData/CustomData/potrait_human.txt')) as f:
        files = f.readlines()

    train_files, val_files = train_val_split(files, 0.1)

    save_dir = 'saved_dir'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if model_mode == 'esp':
        model = SegMatingNet(q=2, p=2)
    else:
        model = CustomNet()

    if pretrained:
        model.load_state_dict(torch.load(pretrained))
        model.train()

    if torch.cuda.is_available():
        model = model.cuda()

    x = torch.randn(1, 3, width, height)
    if torch.cuda.is_available():
        x = x.cuda()
    
    y, matt = model.forward(x)
    g = viz.make_dot(y)
    g.render(save_dir + 'model.png', view=False)
    
    total_paramters = netParams(model)
    print('Total network parameter: ', str(total_paramters))

    criteria = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criteria = criteria.cuda()

    logFileLoc = save_dir + '/log.txt'

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t" % ('Mode','Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
    logger.flush()

    increase_epoches = 50
    max_epoch = 0
    base_lr = 1e-3
    for mode in [TrainMode.SEG, TrainMode.REFINE, TrainMode.SEG_REFINE]:
        model.set_mode(mode)
        base_lr /= 10
        lr = base_lr
        print("-------------------------Mode: %d-----------------------" % (mode))
        #print('lr: ', lr)
        print('max_epoch', max_epoch)

        start_epoch = 0
        max_epoch = max_epoch + increase_epoches 

        if mode == TrainMode.SEG:
            if seg is not None:
                max_epoch = int(seg)

            if start_seg is not None:
                start_epoch = int(start_seg)
            
        elif mode == TrainMode.REFINE:
            if refine is not None:
                max_epoch = int(refine)
            
            if start_refine is not None:
                start_epoch = int(start_refine)
        elif mode == TrainMode.SEG_REFINE:
            if all is not None:
                max_epoch = int(all)

            if start_all is not None:
                start_epoch = int(start_all)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
        # we step the loss by 2 after step size is reached
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=increase_epoches//2, gamma=0.5)

        for epoch in range(start_epoch, max_epoch):
            scheduler.step(epoch)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Learning rate: " +  str(lr))


            transform = CustomTransform(mean=[0.4253, 0.3833, 0.3589], std=[0.2465, 0.2332, 0.2289], 
                                        crop_size=(256, 256), flip=True)
            trainLoader = torch.utils.data.DataLoader(
                MyDataLoader.HumanDataset(train_files, transform),
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )

            valLoader = torch.utils.data.DataLoader(
                MyDataLoader.HumanDataset(val_files, transform),
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
            )

            lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = train({
                'classes': 2,
                'onGPU': torch.cuda.is_available(),
            }, trainLoader, model, criteria, optimizer, epoch, mode)
            
            lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = train({
                'classes': 2,
                'onGPU': torch.cuda.is_available(),
            }, valLoader, model, criteria, optimizer, epoch, mode)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lossTr': lossTr,
                'lossVal': lossVal,
                'iouTr': mIOU_tr,
                'iouVal': mIOU_val,
                'lr': lr
            }, save_dir + '/checkpoint.pth.tar')

            #save the model also
            if (epoch+1) % 1 == 0:
                model_file_name = save_dir + '/' + model_mode + '1_mode_' + str(mode) + '_model_' + str(epoch + 1) + '.pth'
                torch.save(model.state_dict(), model_file_name)


            logger.write("\n%s_%d\t%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (model_mode, mode, epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val))
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode')
    parser.add_argument('--pretrained', dest='pretrained')
    parser.add_argument('--seg', dest='seg')
    parser.add_argument('--refine', dest='refine')
    parser.add_argument('--all', dest='all')
    parser.add_argument('--start_seg', dest='start_seg')
    parser.add_argument('--start_refine', dest='start_refine')
    parser.add_argument('--start_all', dest='start_all')

    args = parser.parse_args()

    main(args.mode, args.pretrained, args.seg, args.refine, args.all, args.start_seg, args.start_refine, args.start_all)