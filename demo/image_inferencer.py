# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import random
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os
import mmrotate  # noqa: F401


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('folder', help='folder with Image file')
    parser.add_argument('num_max', help='maximum number Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-folder', default=None, help='Path to output folder')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def main(args):
    # build the model from a config file and a checkpoint file
    out_folder = args.out_folder
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.folder + args.img)
    # show the results
    show_result_pyplot(
        model,
        args.folder + args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file= out_folder + args.img)


if __name__ == '__main__':
    args = parse_args()
    listafile = os.listdir(args.folder)
    random.shuffle(listafile)
    if(len(listafile)<int(args.num_max)):
        args.num_max = len(listafile)
    

    for i in range(int(args.num_max)):
        args.img = listafile[i]

        main(args)
