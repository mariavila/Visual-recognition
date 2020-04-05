import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='M5-Project')
    parser.add_argument('config', type=str, help='config file to use')
    parser.add_argument('--output', default="output", type=str, help='output folder')
    parser.add_argument('--n_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--data', default="kitti", type=str, choices=["kitti", "mots", "both"], help='data to train with')
    parser.add_argument('--train_only', action="store_true", help='output folder')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--hflip', action='store_true')
    parser.add_argument('--color', action='store_true')

    return parser.parse_args()



