# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zhanghongyuan2017@email.szu.edu.cn

import os
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
import json
from skimage import transform
from skimage import io as sk_io

import warnings
warnings.filterwarnings('ignore')

from utils.model import load_model
from utils.cldetection_utils import check_and_make_dir, calculate_prediction_metrics, visualize_prediction_landmarks

def main(config):
    # GPU device
    gpu_id = config.cuda_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

    # load model
    model = load_model(model_name=config.model_name)
    model.load_state_dict(torch.load(config.load_weight_path, map_location=device))
    model = model.to(device)

    file_list = config.test_txt_path

    with open(file_list,'r') as f:
        data_list = f.readlines()
        data_list = [item.strip() for item in data_list]

    # test result dict
    test_result_dict = {}

    # test mode
    with torch.no_grad():
        model.eval()
        # test all images
        for data_path in tqdm.tqdm(data_list, total=len(data_list)):
            image_path = data_path + '.jpg'
            json_path = data_path + '.json'

            with open(json_path,'r') as f:
                info = json.load(f)
        
            landmarks = []
            for id,i in enumerate(info['shapes']):
                points = i['points'][0]
                landmarks.append(points)

            # load image array
            image = sk_io.imread(image_path)
            h, w = image.shape[:2]
            new_h, new_w = config.image_height, config.image_width

            # preprocessing image for model input
            image = transform.resize(image, (new_h, new_w), mode='constant', preserve_range=True)
            image = image/255
            transpose_image = np.transpose(image, (2, 0, 1))
            torch_image = torch.from_numpy(transpose_image[np.newaxis, :, :, :]).float().to(device)

            # predict heatmap
            heatmap = model(torch_image)

            # transfer to landmarks
            heatmap = np.squeeze(heatmap.cpu().numpy())
            predict_landmarks = []
            for i in range(np.shape(heatmap)[0]):
                landmark_heatmap = heatmap[i, :, :]
                yy, xx = np.where(landmark_heatmap == np.max(landmark_heatmap))
                # there may be multiple maximum positions, and a simple average is performed as the final result
                x0, y0 = np.mean(xx), np.mean(yy)
                # zoom to original image size
                x0, y0 = x0 * w / new_w, y0 * h / new_h
                # append to predict landmarks
                predict_landmarks.append([y0, x0])

            test_result_dict[data_path] = { 'scale': np.asarray([2400,2880]),
                                            'gt': np.asarray(landmarks),
                                            'predict': np.asarray(predict_landmarks)}

    # calculate prediction metrics
    # calculate_prediction_metrics(test_result_dict)

    # visualize prediction landmarks
    if config.save_image:
        check_and_make_dir(config.save_image_dir)
        visualize_prediction_landmarks(test_result_dict, config.save_image_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data parameters | 数据文件路径
    parser.add_argument('--test_txt_path', default='/data/detection/cl-det-fix/test.txt', type=str)

    # model load dir path | 存放模型的文件夹路径
    parser.add_argument('--load_weight_path', default='/home/zt/cl-detection-test/model/checkpoint_epoch_40.pt', type=str)

    # model hyper-parameters: image_width and image_height
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)

    # model test hyper-parameters
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='UNet')

    # result & save
    parser.add_argument('--save_image', type=bool, default=True)
    parser.add_argument('--save_image_dir', type=str, default='/home/zt/cl-detection-test/visualize/')

    experiment_config = parser.parse_args()
    main(experiment_config)
