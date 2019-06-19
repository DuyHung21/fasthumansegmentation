import os
import time 

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage import io, transform

from argparse import ArgumentParser

# from utils.data import preprocess_img
from utils.custom_transforms_final import ToTest
from model.espmatting import SegMatingNet
from model.final2 import CustomNet

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--model_file')
    parser.add_argument('--background')
    parser.add_argument('--time')
    parser.add_argument('--vid')

    args = parser.parse_args()
    demo = None

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    save_vid = args.vid
    if save_vid:
        output = cv2.VideoWriter('output.avi',fourcc, 10.0, (2*w, h))

    get_time = args.time is not None
    if args.mode == 'custom':
        model = CustomNet(get_time)
    elif args.mode == 'esp':
        model = SegMatingNet(p=2, q=2, verbose=get_time)
    else:
        demo = True
        model = torch.load(args.model_file, map_location='cpu')

    background = None
    if args.background is not None:
        background = cv2.imread(args.background)
        background = cv2.resize(background, (w, h))

    if not demo:
        model.set_mode(3)
        model.load_state_dict(torch.load(args.model_file, map_location='cpu'))
    
    model.eval()


    time_seg = []
    time_all = []

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]

        if not demo:
            img_tensor = ToTest()(rgb_frame)
        else:
            img_tensor = ToTest()(rgb_frame, mean=[0.408, 0.439, 0.475], std=[1., 1., 1.])
        img_tensor = torch.unsqueeze(img_tensor, 0)

        begin = time.time()
        if not get_time:
            seg, matt = model(img_tensor)
        else:
            seg, matt, times = model(img_tensor)
            time_seg.append(times[0])
            time_all.append(times[1])

        print(time.time() - begin)

        if background is not None:
            matt = matt[0, 0, :, :].detach().cpu().numpy()
            matt = cv2.resize(matt, (w, h), interpolation=cv2.INTER_CUBIC)
            fg_matt = np.multiply(matt[..., np.newaxis], frame)

            bg_matt = background.copy()
            bg_matt = np.multiply((1 - matt)[..., np.newaxis], bg_matt)

            out_matt = fg_matt + bg_matt
            out_matt[out_matt < 0] = 0
            out_matt[out_matt > 255] = 255
            out_matt = out_matt.astype(np.uint8)

            rs = np.vstack([frame, out_matt])

            if save_vid:
                output.write(rs)



            cv2.imshow('Result', rs)
            # cv2.imshow('Seg', fg_seg)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                name = str(time.time())
                cv2.imwrite('thesis/{}.png'.format(name), rs)
        else:
            matt = matt[0, 0, :, :].detach().cpu().numpy()
            matt = cv2.resize(matt, (w, h), interpolation=cv2.INTER_CUBIC)
            fg_matt = np.multiply(matt[..., np.newaxis], frame)

            fg_matt[fg_matt<0] = 0
            fg_matt[fg_matt>255] = 255
            fg_matt = fg_matt.astype(np.uint8)

            seg = seg[0, :, :, :].detach().cpu().numpy()
            seg = np.argmax(seg, axis=0)
            seg = cv2.resize(seg.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
            fg_seg = np.multiply(seg[..., np.newaxis], frame)

            fg_seg[fg_seg < 0] = 0
            fg_seg[fg_seg > 255] = 255
            fg_seg = fg_seg.astype(np.uint8)
            # cv2.imshow('Image', frame)

            rs = np.hstack([frame, fg_matt])

            if save_vid:
                output.write(rs)

            cv2.imshow('Result', rs)
            # cv2.imshow('Seg', fg_seg)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                name = str(time.time())
                cv2.imwrite('thesis/{}.png'.format(name), fg_seg)
                cv2.imwrite('thesis/{}_img.png'.format(name), frame)

    time_seg = np.array(time_seg)
    time_all = np.array(time_all)
    
    if args.time:
        print ("Time seg: {} +- {}".format(time_seg.mean(), time_seg.std()))
        print ("Time all: {} +- {}".format(time_all.mean(), time_all.std()))

    if args.vid:
        output.release()