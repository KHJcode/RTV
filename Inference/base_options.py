import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--input_video', type=str, help='the path of input video file')
        self.parser.add_argument('--garment_name', type=str, default='example_garment', help='id of the target garment')
        self.parser.add_argument('--output_video', type=str, default='./output.mp4', help='path for the rendered output video')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='directory that stores garment checkpoints')
        self.parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                                 help='device for inference; auto picks cuda when available')
        self.initialized=True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

