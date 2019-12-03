from __future__ import absolute_import

import json
import os
import random
import shutil
import sys

from os.path import join as pj

import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('--img_folder_path', type=str)
parser.add_argument('--valid_json_file', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--lib_path', help='Example: /media/yingges/Data/201910/Deploy/DCN_GPU_cu100/DCN', type=str)
parser.add_argument('--save_img')
parser.add_argument('--display_img', help='Also set this to true if want to save output images.')
args = parser.parse_args()

# args.img_folder_path = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/images'
# args.valid_json_file = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/valid.json'
# args.output_path = output_path = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/augmented_output.json'
# args.lib_path = "/media/yingges/Data/201910/Deploy/DCN_GPU_cu100/DCN"

img_info = None
if arg.valid_json_file is not None:
	gt_json = json.load(open(gt_file))
	img_info = gt_json['images']

test_img_path = [pj(args.img_folder_path, file) for file in os.listdir(args.img_folder_path) if file.endswith('.jpg')]
sys.path.append(os.path.abspath(args.lib_path))
import DCN.fpn.inference as inference

inference.dataset_img_infer(test_img_path, model_path,
							output_path,
							img_info, 'fpn', True, 
							args.display_img, args.save_img,
							32)
