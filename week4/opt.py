import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='M5-Project')
    parser.add_argument('config', type=str, help='config file to use')
    parser.add_argument('--output', default="output", type=str, help='output folder')
    parser.add_argument('--train_only', action="store_true", help='output folder')

    return parser.parse_args()



