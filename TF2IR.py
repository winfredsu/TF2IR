#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import PIL
import json
import os
import sys
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable GPU info

VALID_CONV_ACTS = ['Relu6']
VALID_OPS = ['Conv2D', 'DepthwiseConv2dNative', 'Add', 'Reshape', 'ConcatV2']

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

    def __init__(self, run_dir='.', input_path='graph.pb', img_size=[224,224], input_tensor_name='input', end_points=[], test_image='test.jpg'):
        self.run_dir = run_dir
        self.input_path = input_path
        self.img_size = img_size
        self.input_tensor_name = input_tensor_name
        self.end_points = end_points
        self.net_def = {'layers': []}
        self.test_img = np.array(PIL.Image.open(test_image).resize(self.img_size)).astype(np.float)/128-1
        self.test_img = self.test_img.reshape(1,self.img_size[0],self.img_size[1],3)


    def load_graph(self):
        """
        Load the frozen pb
        """
        f = open(os.path.join(self.run_dir, self.input_path), 'rb')
        self.gd = tf.GraphDef.FromString(f.read())
        tf.import_graph_def(self.gd, name='')

    def traverse_graph(self):
        """
        Find all valid ops in the graph.
        """
        g = tf.get_default_graph()
        input_tensor = g.get_tensor_by_name(self.input_tensor_name)

        ## init the data structures
        # tensors that are already processed
        tensors_ready = [input_tensor]
        # discovered op queue
        ops_discovered = self.__get_real_consumers(input_tensor)
        # dict to store the 'prev_layer' infomation
        info_prev_layers = {}
        for op in ops_discovered:
            info_prev_layers[op.name] = ['input']

        while ops_discovered:
            curr_op = None

            # find the first op in ops_discovered of which all deps are ready
            for op in ops_discovered:
                if set(self.__get_dep_tensors(op)) <= set(tensors_ready):
                    curr_op = op
                    ops_discovered.remove(op)
                    break
            if curr_op is None:
                raise ValueError("Error: ops_discovered is not empty, but no op is ready to be processed")
            
            # init the dict
            print(curr_op.name)
            layer = {'name': curr_op.name}

            # extract informations from curr_op
            if curr_op.type in ['Conv2D', 'DepthwiseConv2dNative']:
                # get ifm/w/b/ofm ops
                ifm_op = self.__get_conv_ifm_op(curr_op)
                w_op = self.__get_conv_w_op(curr_op)
                b_op = self.__get_conv_b_op(curr_op)
                act_type, ofm_op = self.__get_conv_ofm_op(curr_op)

                # conv params
                strides = curr_op.get_attr('strides')[1:3]
                padding = curr_op.get_attr('padding').decode()

                # size params (needs to eval tensors)
                # first we evaluate all tensors
                tensors_to_eval = [op.outputs[0] for op in [ifm_op, w_op, b_op, ofm_op]]
                ifm_data, w_data, b_data, ofm_data = self.__eval_tensors(tensors_to_eval)
                # then get the sizes
                cin = ifm_data.shape[3]
                ifm_size = ifm_data.shape[1:3] # (H,W)
                kernel_size = w_data.shape[:2] # (H,W)
                ofm_size = ofm_data.shape[1:3] # (H,W)
                cout = ofm_data.shape[3]
                # calculate padding sizes (t,b,l,r)
                padding_size = self.__calc_padding_size(padding, strides, ifm_size, kernel_size, ofm_size)

                # quant_params
                ifm_num_bits, ifm_log2scale = self.__get_quant_params_from_op(ifm_op)
                w_num_bits, w_log2scale = self.__get_quant_params_from_op(w_op)
                b_num_bits, b_log2scale = self.__get_quant_params_from_op(b_op)
                ofm_num_bits, ofm_log2scale = self.__get_quant_params_from_op(ofm_op)
                b_shift = ifm_log2scale + w_log2scale - b_log2scale
                assert b_shift >= 0 # TODO: this could be relaxed by forcing the negative b_shift to 0
                ofm_shift = ifm_log2scale + w_log2scale - ofm_log2scale
                assert ofm_shift >= 0

                # calc the quantized ifm/w/b/ofm tensors
                ifm_quant_data = np.squeeze(self.__quant_tensor(ifm_data, ifm_log2scale),0) #HWC
                w_quant_data = self.__quant_tensor(w_data, w_log2scale) # conv2d:HWCiCo, dwconv: HWC1
                b_quant_data = self.__quant_tensor(b_data, b_log2scale) # C
                ofm_quant_data = np.squeeze(self.__quant_tensor(ofm_data, ofm_log2scale),0) # HWC

                # assign output_tensor
                output_tensor = ofm_op.outputs[0]

            elif curr_op.type == 'Add':
                # find x/y/ofm ops
                x_op = self.__get_add_x_op(curr_op)
                y_op = self.__get_add_y_op(curr_op)
                act_type, ofm_op = self.__get_add_ofm_op(curr_op)

                # size params (needs to eval tensors)
                # first we evaluate all tensors
                tensors_to_eval = [op.outputs[0] for op in [x_op, y_op, ofm_op]]
                x_data, y_data, ofm_data = self.__eval_tensors(tensors_to_eval)
                # then get sizes
                assert x_data.shape == y_data.shape
                assert x_data.shape == ofm_data.shape
                cin = x_data.shape[3]
                ifm_size = x_data.shape[1:3] # (H,W)

                # quant_params
                x_num_bits, x_log2scale = self.__get_quant_params_from_op(x_op)
                y_num_bits, y_log2scale = self.__get_quant_params_from_op(y_op)
                ofm_num_bits, ofm_log2scale = self.__get_quant_params_from_op(ofm_op)

                print(x_log2scale, y_log2scale, ofm_log2scale)

                output_tensor = ofm_op.outputs[0]

            elif curr_op.type == 'Reshape':
                # find the output to tensors_ready
                output_tensor = curr_op.outputs[0]

            elif curr_op.type == 'ConcatV2':
                # find the output to tensors_ready
                output_tensor = curr_op.outputs[0]

            else:
                raise ValueError('Unsupported op: ', curr_op)

            # add output tensor to tensors_ready
            tensors_ready.append(output_tensor)
            
            # fill the prev_layers information
            prev_layers = info_prev_layers[curr_op.name]
            layer['prev_layer'] = prev_layers

            # init the next_layers list 
            next_layers = []

            # if the output tensor is not end_point, add all consumers to ops_discovered
            if not (output_tensor.name in self.end_points):
                consumer_ops = self.__get_real_consumers(output_tensor)
                for op in consumer_ops:
                    # add op to next_layers
                    next_layers.append(op.name)
                    # if the op has not been discovered, add the op to ops_discovered and 
                    # init the op in the info_prev_layers list
                    if op not in ops_discovered:
                        ops_discovered.append(op)
                        info_prev_layers[op.name] = [curr_op.name]
                    # if the op has been discovered, update the info_prev_layers
                    else:
                        info_prev_layers[op.name].append(curr_op.name)
            
            # fill the next_layer information
            if next_layers:
                layer['next_layer'] = next_layers
                
            # append the current layer to net_def
            self.net_def['layers'].append(layer)

        # print(self.net_def)

    def __eval_tensors(self, tensors):
        """
        evaluate a list of tensors
        return: a list of numpy array
        """
        g = tf.get_default_graph()
        with tf.Session() as sess:
            return sess.run(tensors, feed_dict={g.get_tensor_by_name(self.input_tensor_name): self.test_img})

    def __get_dep_tensors(self, op):
        """
        (op to tensors)
        return a list of all dependent tensors of this op
        """
        assert op.type in VALID_OPS
        # for conv ops, return the ifm tensor
        if op.type in ['Conv2D', 'DepthwiseConv2dNative']:
            return [self.__get_real_producer_tensor(op.inputs[0])]
        # for add ops, return both x and y
        elif op.type == 'Add':
            assert len(op.inputs) == 2
            return [self.__get_real_producer_tensor(op.inputs[0]), self.__get_real_producer_tensor(op.inputs[1])]
        elif op.type == 'Reshape':
            return [self.__get_real_producer_tensor(op.inputs[0])]
        elif op.type == 'ConcatV2':
            n = op.get_attr('N')
            return [self.__get_real_producer_tensor(op.inputs[i]) for i in range(n)]
    
    def __get_real_producer_tensor(self, tensor):
        """
        (tensor to tensor)
        Get the real producer tensor of a tensor
        """
        op = tensor.op
        while op.type == 'Identity':
            assert len(op.inputs) == 1
            tensor = op.inputs[0]
            op = tensor.op
        return tensor

    def __get_real_producer(self, tensor):
        """
        (tensor to op)
        Get the real producer op of a tensor
        """
        op = tensor.op
        while op.type == 'Identity':
            assert len(op.inputs) == 1
            tensor = op.inputs[0]
            op = tensor.op 
        return op

    def __get_real_consumers(self, tensor):
        """
        (tensor to ops)
        return a list of real consumer ops of a tensor
        """
        ops = tensor.consumers()
        flag = 1
        while flag:
            flag = 0
            for op in ops:
                if op.type == 'Identity':
                    assert len(op.outputs) == 1
                    ops += op.outputs[0].consumers()
                    flag = 1
                    ops.remove(op)
        return ops

    def __get_conv_ifm_op(self, op):
        assert(op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative')
        op_ifm = self.__get_real_producer(op.inputs[0])
        assert op_ifm.type == 'FakeQuantWithMinMaxVars' or 'Placeholder'
        return op_ifm

    def __get_conv_w_op(self, op):
        assert(op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative')
        op_wt = op.inputs[1].op
        assert op_wt.type == 'FakeQuantWithMinMaxVars'
        return op_wt

    def __get_conv_b_op(self, op):
        assert(op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative')
        assert len(op.outputs) == 1
        tensor_conv = op.outputs[0]
        # the bias add op
        assert len(tensor_conv.consumers()) == 1
        op_biasadd = tensor_conv.consumers()[0]
        # the bias tensor
        assert op_biasadd.type == 'Add'
        tensor_bias = op_biasadd.inputs[1]
        op_bias = tensor_bias.op
        assert op_bias.type == 'FakeQuantWithMinMaxVars'
        return op_bias

    def __get_conv_ofm_op(self, op):
        """ 
        Get the output fakequant op from conv2d or dwconv op.

        Returns:
            act_type: type of activation fn (None, Relu, Relu6 ...)
            ofm_op: the output feature map op
        
        The connections after the conv op should be:
        #    CONV   BIAS
        #        \ /
        #        ADD
        #         |
        # (Relu6 or Identities)
        #         |
        #      FakeQuant
        """
        act_type = 'None'
        # the tensor after conv
        assert (op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative')
        assert len(op.outputs) == 1
        tensor_conv = op.outputs[0]
        # the bias add op
        assert len(tensor_conv.consumers()) == 1
        op_biasadd = tensor_conv.consumers()[0]
        # the tensor after bias add
        assert op_biasadd.type == 'Add'
        assert len(op_biasadd.outputs) == 1
        tensor_conv_bias = op_biasadd.outputs[0]
        # find the real consumer(s) of the conv_bias tensor
        # this op should be "relu" or "fakequant"
        ops_found = self.__get_real_consumers(tensor_conv_bias)
        assert len(ops_found) == 1
        # if the found op is fakequant, return it
        if ops_found[0].type == 'FakeQuantWithMinMaxVars':
            pass
        # else if the found op is an activation
        elif ops_found[0].type in VALID_CONV_ACTS:
            act_op = ops_found[0]
            act_type = act_op.type
            assert len(act_op.outputs) == 1
            ops_found = self.__get_real_consumers(act_op.outputs[0])
            assert len(ops_found) == 1
            if ops_found[0].type != 'FakeQuantWithMinMaxVars':
                raise ValueError("Unsupported ocnnection after conv op", op)
        else:
            raise ValueError("Unsupported connection after conv op", op)

        act_quant_op = ops_found[0]
        assert len(act_quant_op.outputs) == 1
        return act_type, act_quant_op

    def __get_add_x_op(self, op):
        assert op.type == 'Add'
        op_x = self.__get_real_producer(op.inputs[0])
        assert op_x.type == 'FakeQuantWithMinMaxVars'
        return op_x
    
    def __get_add_y_op(self, op):
        assert op.type == 'Add'
        op_y = self.__get_real_producer(op.inputs[1])
        assert op_y.type == 'FakeQuantWithMinMaxVars'
        return op_y

    def __get_add_ofm_op(self, op):
        """ 
        Get the output fakequant op from add op.

        Returns:
            act_type: type of activation fn (None, Relu, Relu6 ...)
            ofm_quant_op: the output feature map op
        
        The connections near the ADD op should be:
        #    IFM0   IFM1
        #        \ /
        #        ADD
        #         |
        # (Relu or Identities)
        #         |
        #      FakeQuant
        """   

        act_type = 'None'
        # the tensor after add
        assert (op.type == 'Add')
        assert len(op.outputs) == 1
        tensor_add = op.outputs[0]
        # find the real consumer(s) of the tensor_add tensor
        # this op should be "relu" or "fakequant"
        ops_found = self.__get_real_consumers(tensor_add)
        assert len(ops_found) == 1
        # if the found op is fakequant, return it
        if ops_found[0].type == 'FakeQuantWithMinMaxVars':
            pass
        # else if the found op is an activation
        elif ops_found[0].type in VALID_CONV_ACTS:
            act_op = ops_found[0]
            act_type = act_op.type
            assert len(act_op.outputs) == 1
            ops_found = self.__get_real_consumers(act_op.outputs[0])
            assert len(ops_found) == 1
            if ops_found[0].type != 'FakeQuantWithMinMaxVars':
                raise ValueError("Unsupported ocnnection after conv op", op)
        else:
            raise ValueError("Unsupported connection after conv op", op)

        ofm_quant_op = ops_found[0]
        assert len(ofm_quant_op.outputs) == 1
        return act_type, ofm_quant_op 

    def __get_quant_params_from_op(self, op):
        """
        return num_bits and log2(scaling factor)
        """
        assert op.type == 'FakeQuantWithMinMaxVars' or op.type == 'Placeholder'
        if op.type == 'FakeQuantWithMinMaxVars':
            num_bits = op.get_attr('num_bits')
            min_po2 = op.inputs[1].op.inputs[0].op.get_attr('value').float_val[0]
            log_scale = int(np.log2(np.power(2,num_bits-1)/np.abs(min_po2)))
            return num_bits, log_scale
        elif op.type == 'Placeholder':
            # check if this is the input tensor
            assert op.outputs[0].name == self.input_tensor_name
            return 8, 0

    def __calc_padding_size(self, padding, strides, ifm_size, kernel_size, ofm_size):
        """
        return: [t, b, l, r]

        How to calculate padding size?
        VALID: all 0
        SAME: 
            pads_along_h = max( (ho-1)*s+k-hi, 0 )
            pads_top = pads_along_h // 2
            pads_bot = pads_along_h - pads_top
        """
        # calculate padding sizes
        if padding == 'VALID':
            return [0,0,0,0]
        elif padding == 'SAME':
            pads_along_h = (ofm_size[0]-1)*strides[0]+kernel_size[0]-ifm_size[0]
            pads_along_w = (ofm_size[1]-1)*strides[1]+kernel_size[1]-ifm_size[1]
            tp = int(np.floor(pads_along_h/2))
            bp = int(np.ceil(pads_along_h/2))
            lp = int(np.floor(pads_along_w/2))
            rp = int(np.ceil(pads_along_w/2))
            assert tp>=0 and bp>=0 and lp>=0 and rp>=0
            return [tp,bp,lp,rp]
        else:
            raise ValueError('unsupported padding type: ', padding)

    def __quant_tensor(self, tensor, scale):
        return (np.power(2,scale)*tensor).astype(np.int32)

    def __add_rounding_to_bias_and_truncate(self, b_quantized, b_ls_raw, ofm_rs):
        """
        add half of OFM LSB to bias, so that the hardware can only perform shifting
        and truncate the bias if the b_ls_raw is negative

        returns:
        b, b_shift
        """
        shift = ofm_rs-1-b_ls_raw
        if shift >= 0:
            b_quantized += (1<<shift)

        if b_ls_raw < 0:
            np.right_shift(b_quantized,-b_ls_raw)
            return b_quantized, 0
        else:
            return b_quantized, b_ls_raw

    def test(self):
        g = tf.get_default_graph()
        for op in g.get_operations():
            if op.name == 'FeatureExtractor/MobilenetV2/layer_19_2_Conv2d_2_3x3_s2_384/Conv2D_Fold':
                curr_op = op

                # get ifm/w/b/ofm ops
                ifm_op = self.__get_conv_ifm_op(curr_op)
                w_op = self.__get_conv_w_op(curr_op)
                b_op = self.__get_conv_b_op(curr_op)
                act_type, ofm_op = self.__get_conv_ofm_op(curr_op)

                # size params (needs to eval tensors)
                # first we evaluate all tensors
                tensors_to_eval = [op.outputs[0] for op in [ifm_op, w_op, b_op, ofm_op]]
                ifm_data, w_data, b_data, ofm_data = self.__eval_tensors(tensors_to_eval)

                print(w_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract net arch and raw params from quant-aware trained frozen graph')
    parser.add_argument('-d', '--directory', default='.', help='running directory')
    parser.add_argument('-i', '--input_path', default='graph.pb', help='input frozen pb')
    parser.add_argument('--image_height', type=int, default=160, help='input image height')
    parser.add_argument('--image_width',  type=int, default=160, help='input image width')
    parser.add_argument('--input_tensor_name', default='normalized_input_image_tensor:0', help='input tensor name')
    parser.add_argument('--end_points', default=['concat_1:0', 'concat:0'], nargs='+', help='final point names')
    parser.add_argument('--test_image', default='test.jpg', help='example input image for evaluating fmaps')
    args = parser.parse_args()

    tf2ir = TF2IR(args.directory, args.input_path, [args.image_height, args.image_width], args.input_tensor_name, args.end_points, args.test_image)
    tf2ir.load_graph()
    tf2ir.traverse_graph()
    # tf2ir.test()

