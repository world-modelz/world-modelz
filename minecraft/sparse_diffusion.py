import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    # plan:
    # load 64 frames from mine rl-dataset
    # load decoder model
    # quantize data
    # write 3d index sampler
    # run single batch training


if __name__ == '__main__':
    main()
