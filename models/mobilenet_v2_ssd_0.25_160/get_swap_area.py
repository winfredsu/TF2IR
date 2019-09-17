#!/usr/bin/env python

import numpy as np
import json
import os
import shutil
import sys
import argparse
from operator import mul
from functools import reduce

def get_swap_area(input_path, output_path):
    # load the converted net definition
    layers = json.load(open(input_path, 'r'))
    # record the max_tensor_size
    # Note: for ASIC, the max_tensor_size should be calculated in line with the in-memory data format
    max_tensor_size = layers[0]['input_channel_num'] * layers[0]['input_size']['height'] * layers[0]['input_size']['width']
    max_sized_tensor_name = 'input'
    # record the visited tensors, value indicates how many times it will be used
    visited_tensors = {'input': 1}
    # record the number of tensors that must be maintained
    maintained_tensor_count = [0]*len(layers)
    
    for i, layer in enumerate(layers):
        # update the max_tensor_size
        curr_tensor_size = get_output_tensor_size(layer)
        if curr_tensor_size > max_tensor_size:
            max_tensor_size = curr_tensor_size
            max_sized_tensor_name = layer['name']
        # add layer output to visted_tensors
        visited_tensors[layer['name']] = len(layer['next_layer'])
        # find the number of visited tensors which have not been consumed
        maintained_tensor_count[i] = sum([visited_tensors[t]!=0 for t in list(visited_tensors.keys())])
        # consume the input tensor(s)
        for prev in layer['prev_layer']:
            visited_tensors[prev] -= 1

    print('number of layers =', len(layers))
    print('max tensor size =', max_tensor_size, ' at layer ', max_sized_tensor_name)
    print('max number of tensors to be maintained =', max(maintained_tensor_count))

    swap_info = {'max_tensor_size': max_tensor_size, 'num_swap_area': max(maintained_tensor_count)}
    f = open(output_path, 'w')
    f.write(json.dumps(swap_info, indent=2))
    f.close()      

def get_output_tensor_size(layer):
    """
    input: layer object
    output: output tensor size
    
    Note: for ASIC, the size should be calculated in line with the in-memory data format
    """
    if layer['operation'] in ['conv', 'dwconv']:
        return layer['output_size']['height']*layer['output_size']['width']*layer['output_channel_num']
    elif layer['operation'] == 'add':
        return layer['input_size']['height']*layer['input_size']['width']*layer['input_channel_num']
    elif layer['operation'] == 'reshape':
        return reduce(mul, layer['output_shape'])
    else:
        raise ValueError('Unsupproted layer type: ', layer['operation'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Determine the maximum feature map and number of swap area')
    parser.add_argument('-i', '--input_path', default='net_def.json', help='input net_def file')
    parser.add_argument('-o', '--output_path', default='swap_info.json', help='output json file')
    args = parser.parse_args()

    get_swap_area(args.input_path, args.output_path)
