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
    - Mobilenet-V2-SSD

    Valid ops:

    Conversion rules:
    - Reshape

    """

    def __init__(self, run_dir='.', input_path='graph.pb', img_size=[224,224], input_tensor_name='input'):
        self.run_dir = run_dir
        self.input_path = input_path
        self.img_size = img_size
        self.input_tensor_name = input_tensor_name

        self.net_def = {'layers': []}

    def load_graph(self):
        """
        Load the frozen pb
        """
        f = open(os.path.join(self.run_dir, self.input_path), 'rb')
        self.gd = tf.GraphDef.FromString(f.read())
        tf.import_graph_def(self.gd, name='')

    def get_valid_ops(self):
        """
        Find all valid ops in the graph. 
        Valid ops includes:
        - Conv2D
        - DWConv
        """
        g = tf.get_default_graph()
        for op in g.get_operations():
            if op.type == 'Conv2D':
                op_def = {}
                op_def['name'] = op.name
                op_def['type'] = op.type
                op_def['params'] = {}
                op_def['params']['strides'] = op.get_attr('strides')
                op_def['params']['padding'] = op.get_attr('padding').decode()
                op_def['tensors'] = {}
                # get the inputs of the conv op
                


                op_def['prev_layers'] = []
                op_def['next_layers'] = []

                

                print(op.get_attr('padding').decode())
                break

    def __get_producer(self, tensor):
        """
        Get the "real" producer op of the current tensor
        """
        op = tensor.op
        while op.type == 'Identity':
            tensor = op.inputs[0]
            op = tensor.op 
        return op

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract net arch and raw params from quant-aware trained frozen graph')
    parser.add_argument('-d', '--directory', default='.', help='running directory')
    parser.add_argument('-i', '--input_path', default='graph.pb', help='input frozen pb')
    parser.add_argument('--image_height', type=int, default=300, help='input image height')
    parser.add_argument('--image_width',  type=int, default=300, help='input image width')
    parser.add_argument('--input_tensor_name', default='normalized_input_image_tensor', help='input tensor name')
    args = parser.parse_args()

    tf2ir = TF2IR(args.directory, args.input_path, [args.image_height, args.image_width], args.input_tensor_name)
    tf2ir.load_graph()
    tf2ir.get_valid_ops()

