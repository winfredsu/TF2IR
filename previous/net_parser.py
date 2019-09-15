#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import json
import os
import sys
import argparse

class NetParser(object):
    """
    Load a frozen graph, export net definitions (of conv layers) to net_def.json, 
    and dump raw params (conv and dense) to a raw_params.npy. 

    This parser only works for sequential networks. 

    For object detection networks, the detection heads are leaved for patch_obj_det
    """

    def __init__(self, run_dir='.', img_size=[160, 160], input_path='graph.pb', output_path='net_def.json', is_obj_det_net=False):
        self.run_dir = run_dir
        self.input_path = input_path
        self.net_def = {'layers': []}
        self.output_path = output_path
        self.img_size = img_size
        self.is_obj_det_net = is_obj_det_net
        self.params = {}
        self.conv_param_size = 0
        self.dense_param_size = 0
        self.max_fmap_size = 0 

        if is_obj_det_net:
            self.input_name = 'normalized_input_image_tensor'
        else:
            self.input_name = 'input'

    def parse(self):
        """
        Extract params from graph to self.net_def.
        """
        self.__load_graph()
        self.__get_layers()
        self.__add_layer_params()
        self.export_net_def()
        self.export_params()
        self.export_config()

    def export_net_def(self):
        """
        Dump net_def to json
        """
        f = open(os.path.join(self.run_dir, self.output_path), 'w')
        f.write(json.dumps(self.net_def, indent=2))
        f.close()

    def export_params(self):
        np.save(os.path.join(self.run_dir, 'raw_params.npy'), self.params)

    def export_config(self):
        config = {'sdram_map': 
                    {'param_ost':0,'buf0_ost':int(self.conv_param_size),'buf1_ost':int(self.conv_param_size+self.max_fmap_size),'sdram_used':int(self.conv_param_size+2*self.max_fmap_size)},
                 } 
        print(config)
        f = open(os.path.join(self.run_dir, 'config.json'), 'w')
        f.write(json.dumps(config, indent=2))
        f.close()
        
        print('########### SUMMARY #############')
        print('Conv Params ', self.conv_param_size, ' byte')
        print('Dense Params ', self.dense_param_size, ' byte')
        print('Max FMAP Size ', self.max_fmap_size, ' byte')
        print('#################################')


    def __load_graph(self):
        self.gd = tf.GraphDef.FromString(open(os.path.join(self.run_dir, self.input_path),'rb').read())
        tf.import_graph_def(self.gd, name='')

    def __find_tensor_by_keys(self, keys):
        """
        Return the tensor if all key in keys are in the tensorname
        params:
            keys: a list of keywords
        return:
            the found tensor
        """
        for tensor in tf.get_default_graph().as_graph_def().node:
            # if all keys appear in the nodename, return the node
            if all(key in tensor.name for key in keys):
                return tensor
        raise ValueError('no such tensor! keys: '+str(keys))

    def __find_tensor_by_name(self, name):
        """
        Return the tensor named 'name'
        """
        for tensor in tf.get_default_graph().as_graph_def().node:
            if tensor.name == name:
                return tensor
        raise ValueError('no tensor named '+name)

    def __get_layers(self, keys=['Conv2D', 'DepthwiseConv2dNative']):
        """
        Get padding, stride, opname and related i/o/w/b tensor names of primary layer.
        This only works for sequential networks.
        
        params:
            keys: list of opnames of primary layer

        return:
            None
        """
        for tensor in tf.get_default_graph().as_graph_def().node:
            for key in keys:
                # if the current tensor is a primary layer
                if tensor.op == key:

                    if 'WeightSharedConvolutionalBoxPredictor' in tensor.name:
                        continue

                    print(tensor.name)

                    # save basic layer params and i/w/b/o tensor names
                    layer_def = {}
                    layer_def['id'] = tensor.name.split('/')[-2]              
                    layer_def['target'] = 'engine'
                    layer_def['layer_params'] = {}
                    layer_def['tensors'] = {}

                    layer_def['layer_params']['op'] = tensor.op
                    layer_def['layer_params']['activation'] = 'relu6'
                    layer_def['layer_params']['stride'] = tensor.attr['strides'].list.i[1]
                    layer_def['layer_params']['padding'] = tensor.attr['padding'].s.decode()

                    layer_def['tensors']['ifm'] = tensor.input[0]
                    layer_def['tensors']['w']   = tensor.input[1]
                    layer_def['tensors']['ofm'] = self.__find_tensor_by_keys([layer_def['id'], 'act_quant/FakeQuantWithMinMaxVars']).name

                    # the final dense layer has no bias_quant
                    if '1x1' in tensor.name or 'Dense' in tensor.name:
                        layer_def['target'] = 'mcu'
                        layer_def['tensors']['b'] = self.__find_tensor_by_keys([layer_def['id'], 'biases/read']).name
                    else:
                        layer_def['tensors']['b'] = self.__find_tensor_by_keys([layer_def['id'], 'bias_quant/FakeQuantWithMinMaxVars']).name

                    self.net_def['layers'].append(layer_def)

    def __eval_layer_tensors(self, tensors):
        inp, ifm, w, b, ofm = tf.import_graph_def(self.gd, return_elements=[
            self.input_name+':0', tensors['ifm']+':0', tensors['w']+':0', tensors['b']+':0', tensors['ofm']+':0'])
        with tf.Session(graph=inp.graph) as sess:
            ifm, w, b, ofm = sess.run([ifm,w,b,ofm], feed_dict={inp: np.zeros([1,self.img_size[0],self.img_size[1],3])})
        return ifm, w, b, ofm

    def __get_quant_params(self, tensorname):
        """
        Return number of quantization bits and the log2 scale factor.
        Note that for the input tensor, the bit_num is always 8 and scale is always 7.
        This requires the input range is in [-1,1)
        """
        if tensorname == self.input_name:
            num_bits = 8 
            scale = 7 
            return num_bits, scale
        else:
            num_bits = self.__find_tensor_by_name(tensorname).attr['num_bits'].i
            min_po2 = self.__find_tensor_by_name(tensorname.replace('FakeQuantWithMinMaxVars', 'min_po2')).attr['value'].tensor.float_val[0]
            scale = int(np.log2(np.exp2(num_bits-1)/np.abs(min_po2)))
            return num_bits, scale 

    def __add_layer_idx(self):
        for i in range(len(self.net_def['layers'])):
            self.net_def['layers'][i]['idx'] = i

    def __add_layer_params(self):
        """
        Add more layer params to layer_def, including tensor shapes and quant params. 
        """
        for layer_def in self.net_def['layers']:    
            # extract the tensors  
            ifm, w, b, ofm = self.__eval_layer_tensors(layer_def['tensors'])
            
            # save w and b to self.params
            self.params[layer_def['id']] = {'w': w, 'b': b}

            # get tensor shapes
            ifm_shape = ifm.shape
            w_shape   = w.shape
            ofm_shape = ofm.shape
            b_shape   = b.shape
            
            layer_def['layer_params']['hi'] = ifm_shape[1]
            layer_def['layer_params']['wi'] = ifm_shape[2]
            layer_def['layer_params']['ho'] = ofm_shape[1]
            layer_def['layer_params']['wo'] = ofm_shape[2]
            layer_def['layer_params']['ci'] = ifm_shape[3]
            layer_def['layer_params']['co'] = ofm_shape[3]
            layer_def['layer_params']['k']  = w_shape[0] 

            # update max_fmap_size. Here we should use the aligned mem size
            fmap_size = int(ifm_shape[1]*ifm_shape[3]*np.ceil(ifm_shape[2]/16.0)*16)
            if fmap_size > self.max_fmap_size:
                self.max_fmap_size = fmap_size

            # for the dense layer, biases are not quantized
            if '1x1' in layer_def['id'] or 'Dense' in layer_def['id']:
                # extract the quantization parameters
                ifm_num_bits = 8
                ifm_scale = 4
                layer_def['layer_params']['ifm_num_bits'] = ifm_num_bits
                layer_def['layer_params']['ifm_scale'] = ifm_scale

                w_num_bits, w_scale = self.__get_quant_params(layer_def['tensors']['w'])
                layer_def['layer_params']['w_num_bits'] = w_num_bits
                layer_def['layer_params']['w_scale'] = w_scale

                # for the dense layer, bias is non-scaled float, so the left shift number is ifm_scale + w_scale
                b_ls_raw = ifm_scale+w_scale
                layer_def['layer_params']['b_ls_raw'] = b_ls_raw

                # accumulate to dense layer param size
                self.dense_param_size += int(np.prod(w_shape)*w_num_bits/8 + np.prod(b_shape)*32/8)

            else:
                # extract the quantization parameters
                ifm_num_bits, ifm_scale = self.__get_quant_params(layer_def['tensors']['ifm'])
                layer_def['layer_params']['ifm_num_bits'] = ifm_num_bits
                layer_def['layer_params']['ifm_scale'] = ifm_scale

                w_num_bits, w_scale = self.__get_quant_params(layer_def['tensors']['w'])
                layer_def['layer_params']['w_num_bits'] = w_num_bits
                layer_def['layer_params']['w_scale'] = w_scale

                b_num_bits, b_scale = self.__get_quant_params(layer_def['tensors']['b'])
                layer_def['layer_params']['b_num_bits'] = b_num_bits
                layer_def['layer_params']['b_scale'] = b_scale

                ofm_num_bits, ofm_scale = self.__get_quant_params(layer_def['tensors']['ofm'])
                layer_def['layer_params']['ofm_num_bits'] = ofm_num_bits
                layer_def['layer_params']['ofm_scale'] = ofm_scale

                b_ls_raw = ifm_scale+w_scale-b_scale
                layer_def['layer_params']['b_ls_raw'] = b_ls_raw
                # ensure b_ls is non-negative. If b_ls_raw is negative, just crop the LSBs of bias.
                if b_ls_raw >= 0:
                    layer_def['layer_params']['b_ls'] = b_ls_raw
                else:
                    layer_def['layer_params']['b_ls'] = 0
                ofm_rs = ifm_scale+w_scale-ofm_scale
                if ofm_rs < 0:
                    raise ValueError('negative ofm right shift in layer '+layer_def['id'])
                layer_def['layer_params']['ofm_rs'] = ofm_rs

                # accumulate to conv layer param size
                # for the dwconv layers, the actual mem size is 4/3 np.prod(w_shape)
                if layer_def['layer_params']['op'] == 'DepthwiseConv2dNative':
                    alpha = 4/3.0
                else:
                    alpha = 1.0
                self.conv_param_size += int(alpha*np.prod(w_shape)*w_num_bits/8 + np.prod(b_shape)*b_num_bits/8)
            
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='convert input frozen graph to net_def.json')
    parser.add_argument('-d', '--directory', default='.', help='running directory')
    parser.add_argument('-i', '--input_path', default='frozen.pb', help='input frozen pb')
    parser.add_argument('--image_height', type=int, default=160, help='input image height')
    parser.add_argument('--image_width',  type=int, default=160, help='input image width')
    parser.add_argument('--is_obj_det_net', type=bool, default=False, help='is object detection network')
    args = parser.parse_args()

    parser = NetParser(args.directory, [args.image_height, args.image_width], args.input_path, 'net_def.json', args.is_obj_det_net)
    parser.parse()