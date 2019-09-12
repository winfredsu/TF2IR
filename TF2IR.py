#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import json
import os
import sys
import argparse

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

    def __init__(self, run_dir='.', input_path='graph.pb', img_size=[224,224], input_tensor_name='input', end_points=[]):
        self.run_dir = run_dir
        self.input_path = input_path
        self.img_size = img_size
        self.input_tensor_name = input_tensor_name
        self.end_points = end_points
        self.net_def = {'layers': []}


    def load_graph(self):
        """
        Load the frozen pb
        """
        f = open(os.path.join(self.run_dir, self.input_path), 'rb')
        self.gd = tf.GraphDef.FromString(f.read())
        tf.import_graph_def(self.gd, name='')

    def traverse_graph(self):
        """
        Find all valid ops in the graph by BFS.
        Valid ops includes:
        - Conv2D
        - DWConv
        """
        g = tf.get_default_graph()
        input_tensor = g.get_tensor_by_name(self.input_tensor_name)

        tensors_ready = [input_tensor]
        ops_discovered = self.__get_real_consumers(input_tensor)

        while ops_discovered:
            curr_op = None
            # find the first op in ops_discovered of which all deps are ready
            for op in ops_discovered:
                if set(self.__get_dep_tensors(op)) <= set(tensors_ready):
                    curr_op = op
                    # print('remove ', curr_op.name)
                    # print('before remove ', len(ops_discovered))
                    ops_discovered.remove(op)
                    # print('after remove ', len(ops_discovered))                    
                    break
            if curr_op is None:
                raise ValueError("Error: ops_discovered is not empty, but no op is ready to be processed")

            # extract informations from curr_op
            if curr_op.type in ['Conv2D', 'DepthwiseConv2dNative']:
                # find the ofm op and add its ouptut to tensors_ready
                act_type, ofm_op = self.__get_conv_ofm_op(curr_op)
                ofm_tensor = ofm_op.outputs[0]
                tensors_ready.append(ofm_tensor)
                
                # find all consumers of the ofm_tensor and add to ops_discovered
                if not ofm_tensor.name in self.end_points:
                    consumer_ops = self.__get_real_consumers(ofm_tensor)
                    print(ofm_tensor.name)
                    ops_discovered += consumer_ops
                    # print('add ', [op.name for op in consumer_ops])

            elif curr_op.type == 'Add':
                # find the ofm op and add it to tensors_ready
                act_type, ofm_op = self.__get_add_ofm_op(curr_op)
                ofm_tensor = ofm_op.outputs[0]
                tensors_ready.append(ofm_tensor)

                # find all consumers of the ofm_tensor and add to ops_discovered
                if not ofm_tensor.name in self.end_points:
                    consumer_ops = self.__get_real_consumers(ofm_tensor)
                    ops_discovered += consumer_ops
                    # print('add ', [op.name for op in consumer_ops])

            elif curr_op.type == 'Reshape':
                # find the output to tensors_ready
                output_tensor = curr_op.outputs[0]
                tensors_ready.append(output_tensor)

                # find all consumers and add to ops_discovered
                if not output_tensor.name in self.end_points:
                    consumer_ops = self.__get_real_consumers(ofm_tensor)
                    ops_discovered += consumer_ops

            elif curr_op.type == 'ConcatV2':
                # find the output to tensors_ready
                output_tensor = curr_op.outputs[0]
                tensors_ready.append(output_tensor)

                # find all consumers and add to ops_discovered
                if not output_tensor.name in self.end_points:
                    consumer_ops = self.__get_real_consumers(ofm_tensor)
                    ops_discovered += consumer_ops

            else:
                raise ValueError('Unsupported op: ', curr_op)

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
            print(tensor.name)
            print([op.name for op in ops])
            print('\n')
            flag = 0
            for op in ops:
                if op.type == 'Identity':
                    assert len(op.outputs) == 1
                    ops += op.outputs[0].consumers()
                    flag = 1
                    ops.remove(op)
        return ops

    def __get_conv_w_tensor(self, op):
        assert(op.type == 'Conv2D' or op.type == 'DepthwiseConv2dNative')
        return op.inputs[1]

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
        return num_bits and min_po2 (negative)
        """
        assert op.type == 'FakeQuantWithMinMaxVars'
        num_bits = op.get_attr('num_bits')
        min_po2 = op.inputs[1].op.inputs[0].op.get_attr('value').float_val[0]
        return num_bits, min_po2


    def test(self):
        g = tf.get_default_graph()
        # for op in g.get_operations():
        #     if op.type == 'ConcatV2':
        #         print(op.get_attr('N'))
        #         break

        op = g.get_operation_by_name('FeatureExtractor/MobilenetV2/expanded_conv/input')
        print(len(op.outputs[0].consumers()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract net arch and raw params from quant-aware trained frozen graph')
    parser.add_argument('-d', '--directory', default='.', help='running directory')
    parser.add_argument('-i', '--input_path', default='graph.pb', help='input frozen pb')
    parser.add_argument('--image_height', type=int, default=300, help='input image height')
    parser.add_argument('--image_width',  type=int, default=300, help='input image width')
    parser.add_argument('--input_tensor_name', default='normalized_input_image_tensor:0', help='input tensor name')
    parser.add_argument('--end_points', default=['concat_1', 'concat'], nargs='+', help='final point names')
    args = parser.parse_args()

    tf2ir = TF2IR(args.directory, args.input_path, [args.image_height, args.image_width], args.input_tensor_name, args.end_points)
    tf2ir.load_graph()
    # tf2ir.traverse_graph()
    tf2ir.test()

