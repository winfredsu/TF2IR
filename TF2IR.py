#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import json
import os
import sys
import argparse

class TF2IR(object):
    """
    Load a quant-aware trained frozen graph (in .pb), output
    a json file and a file containing raw params.

    Currently supported networks:
    - Mobilenet-V1
    - Mobilenet-V2-SSD

    """

    def __init__(self, run_dir='.', input_path='graph.pb', img_size=[224,224], input_tensor_name='input'):
        self.run_dir = run_dir
        self.input_path = input_path
        self.img_size = img_size
        self.input_tensor_name = input_tensor_name

    def load_graph(self):
        """
        Load the frozen pb
        """
        f = open(os.path.join(self.run_dir, self.input_path), 'rb')
        self.gd = tf.GraphDef.FromString(f.read())
        tf.import_graph_def(self.gd, name='')

    def find_all_valid_ops(self):
        """
        Find all valid ops in the graph. 
        Valid ops includes:
        - Conv2D
        - DWConv
        """
        for node in tf.get_default_graph().as_graph_def().node:
            if 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract net arch and raw params from quant-aware trained frozen graph')
    parser.add_argument('-d', '--directory', default='.', help='running directory')
    parser.add_argument('-i', '--input_path', default='frozen.pb', help='input frozen pb')
    parser.add_argument('--image_height', type=int, default=200, help='input image height')
    parser.add_argument('--image_width',  type=int, default=200, help='input image width')
    parser.add_argument('--input_tensor_name', default='input', help='input tensor name')
    args = parser.parse_args()

    parser = NetParser(args.directory, [args.image_height, args.image_width], args.input_path, args.input_tensor_name)
    parser.parse()
    