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
parser.add_argument('--model_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--lib_path', help='Example: /media/yingges/Data/201910/Deploy/DCN_GPU_cu100/DCN', type=str)
parser.add_argument('--save_img', default=False, action='store_true')
parser.add_argument('--display_img', default=False, action='store_true', help='Also set this to true if want to save output images.')
parser.add_argument('--finegrained_cls', default=False, action='store_true')
parser.add_argument('--confidence_thr', default=1e-3, type=float)
args = parser.parse_args()

# args.img_folder_path = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/og_files/images'
# args.valid_json_file = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/og_files/fg_valid_sizethr625.json'
# args.output_path = '/media/yingges/Data/201910/FT/FTData/ft_det_cleanedup/ignore_toosmall/11_30/og_files/fg_output_sizethr625.json'
# args.lib_path = "/home/yingges/experiment/DCN/DCN_yx_test"
args.lib_path = '.'

img_info = None
if args.valid_json_file is not None:
	gt_json = json.load(open(args.valid_json_file))
	img_info = gt_json['images']

test_img_path = [pj(args.img_folder_path, file) for file in os.listdir(args.img_folder_path) if file.endswith('.jpg')]
sys.path.append(os.path.abspath(args.lib_path))
# if args.finegrained_cls:
# 	import DCN.fpn.inference as inference
# 	model_type = 'fpn'
	# model_path = 'DCN/model/fg_epoch21/rfcn_dcn_voc'
# else:
# 	import DCN.rfcn.inference as inference
# 	model_type = 'rfcn'
# 	model_path = 'DCN/model/generic/rfcn_dcn_voc'
import DCN.fpn.inference as inference
model_type = 'fpn'
model_path = args.model_path

inference.dataset_img_infer(image_names=test_img_path,
							model_path=model_path, 
							json_file=args.output_path, 
							image_info_list=img_info,
							model_type=model_type,
							cuda_provided=True,
							display=args.display_img,
							save_img=args.save_img,
							batch=32,
							thresh=args.confidence_thr,
							fg_cls=args.finegrained_cls)
