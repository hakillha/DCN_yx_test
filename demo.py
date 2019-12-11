from __future__ import absolute_import
import argparse
import sys
sys.path.insert(0, '/media/yingges/Data/201910/Experiments/Test01/PG02/DCN/DCN_yx_test')

import DCN.fpn.inference as inference

parser = argparse.ArgumentParser('')
parser.add_argument('--img_path', type=str)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

print(inference.batch_img_infer([args.img_path],
								args.model_path,
								'fpn'))