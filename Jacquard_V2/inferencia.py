import argparse
import logging
import os
import torch.utils.data
from models.ggcnn import GGCNN
from models.common import post_process_output
from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset
from matplotlib import pyplot as plt
import numpy as np
from utils.dataset_processing import grasp, image

from utils.dataset_processing.grasp import my_plot_detect_grasps

logging.basicConfig(level=logging.INFO)

root_dir =  "C:/TFM/Jacquard_V2/Jacquard_V2/result_test"

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate mobilenet')
    # Network
    parser.add_argument('--network', type=str, default='/home/lqh/ggcnn/my_weights_jacquard/epoch_98_iou_0.98',
                        help='Path to saved network to evaluate')
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default='jacquard', help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, default='/data_jiang/lqh/J_11', help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=0, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.1, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0, help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=1, help='Dataset workers')
    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', default=1, help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true', help='Jacquard-dataset style output')
    parser.add_argument('--vis', default=1, help='Visualise the network output')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


def numpy_to_torch(s):
    if len(s.shape) == 2:
        prueba = torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        return torch.from_numpy(np.expand_dims(prueba, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))
            
def get_depth(depth_file, rot=0, zoom=1.0, output_size=300,):
    depth_img = image.DepthImage.from_tiff(depth_file)
    depth_img.rotate(rot)
    depth_img.normalise()
    depth_img.zoom(zoom)
    depth_img.resize((output_size, output_size))
    return depth_img.img

def get_rgb(rgb_file, rot=0, zoom=1.0, normalise=True, output_size=300,):
    rgb_img = image.Image.from_file(rgb_file)
    rgb_img.rotate(rot)
    rgb_img.zoom(zoom)
    #rgb_img.resize(output_size, output_size)
    if normalise:
        rgb_img.normalise()
        rgb_img.img = rgb_img.img.transpose((2, 0, 1))
    return rgb_img.img


if __name__ == '__main__':
    args = parse_args()

    # Load Network
    net = torch.load(args.network, weights_only=False)
    device = torch.device("cuda:0")

    # Load Dataset
    #logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    #Dataset = get_dataset(args.dataset)
    #test_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
    #                       random_rotate=args.augment, random_zoom=args.augment,
    #                       include_depth=args.use_depth, include_rgb=args.use_rgb)
    #test_data = torch.utils.data.DataLoader(
    #    test_dataset,
    #    batch_size=1,
    #    shuffle=False,
    #    num_workers=args.num_workers
    #)
    #logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    if args.jacquard_output:
        jo_fn = args.network + '_jacquard_output.txt'
        with open(jo_fn, 'w') as f:
            pass

    num1 = []
    num2 = []
    s = 0
    folder_path_false = root_dir + '/false_image'
    folder_path_false2 = root_dir + '/false_image2'
    folder_path_true = root_dir + '/true_image'
    if not os.path.exists(folder_path_false):
        os.makedirs(folder_path_false)
    if not os.path.exists(folder_path_false2):
        os.makedirs(folder_path_false2)
    if not os.path.exists(folder_path_true):
        os.makedirs(folder_path_true)

    include_depth = args.use_depth
    include_rgb = args.use_rgb

    depth_img = get_depth(os.path.join("C:/Users/jpg/Downloads/imagen_depth.tiff"))
    rgb_img = get_rgb(os.path.join("C:/Users/jpg/Downloads/imagen_rgb.png"))

    if include_depth and include_rgb:
        x = numpy_to_torch(
            np.concatenate(
                (np.expand_dims(depth_img, 0),
                rgb_img),0
                )
            )
    elif include_depth:
        x = numpy_to_torch(depth_img)
    elif include_rgb:
        x = numpy_to_torch(rgb_img)


    with torch.no_grad():
        xc = x.to(device)
        pred = net.predict(xc)
        print(pred)

        q_img, ang_img, width_img = post_process_output(pred['pred']['pos'], pred['pred']['cos'],
                                                            pred['pred']['sin'], pred['pred']['width'])
        with open(folder_path_true + '/document.txt', 'a') as f:
            f.write("q:" + str(q_img[0]))
            f.write("\n")
            f.write("ang:" + str(ang_img[0]))
            f.write("\n")
            f.write("width:" + str(width_img[0]))
            f.write("\n")

        evaluation.plot_output_2(rgb_img, depth_img, q_img,
                                       ang_img, no_grasps=args.n_grasps, grasp_width_img=width_img
                                       )
        file_name_true = 'prueba.png'
        plt.savefig(os.path.join(folder_path_true, file_name_true))

       