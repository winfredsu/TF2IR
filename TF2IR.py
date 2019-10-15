#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import PIL
import json
import os
import shutil
import sys
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable GPU info

VALID_ACTS = ['Relu6']
VALID_CONV_OPS = ['Conv2D', 'DepthwiseConv2dNative']
VALID_OPS = ['Conv2D', 'DepthwiseConv2dNative', 'Add', 'Reshape', 'AvgPool']

class TF2IR(object):
    """
    Load a quant-aware trained frozen graph (in .pb), output
    a json file and a file containing raw params.

    Currently supported networks:
    - Mobilenet-V2-SSD

    """

    def __init__(self, input_path='graph.pb', output_path='.', img_size=[224,224], input_tensor_name='input', end_points=[], test_image='test.jpg'):
        self.output_path = output_path
        self.input_path = input_path
        self.img_size = img_size
        self.input_tensor_name = input_tensor_name
        self.end_points = end_points
        self.net_def = {'layers': []}
        # note that PIL.resize() requires [W,H] as param
        self.test_img = np.array(PIL.Image.open(test_image).resize([self.img_size[1], self.img_size[0]])).astype(np.float)/128-1
        self.test_img = self.test_img.reshape(1,self.img_size[0],self.img_size[1],3)

        self.output_param_path = os.path.join(output_path, 'params')            
        self.output_netdef_path = os.path.join(output_path, 'net_def.json')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        if os.path.exists(self.output_param_path):
            shutil.rmtree(self.output_param_path)
        os.makedirs(self.output_param_path)

    def load_graph(self):
        """
        Load the frozen pb
        """
        f = open(self.input_path, 'rb')
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
            # for op in ops_discovered:
            for op in reversed(ops_discovered):
                if set(self.__get_dep_tensors(op)) <= set(tensors_ready):
                    curr_op = op
                    ops_discovered.remove(op)
                    break
            if curr_op is None:
                raise ValueError("Error: ops_discovered is not empty, but no op is ready to be processed")
            
            # init the dict
            print(curr_op.name)
            layer = {'name': curr_op.name.replace('/','_')}

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

                # fill layer dict
                if curr_op.type == 'Conv2D':
                    layer['operation'] = 'conv'
                elif curr_op.type == 'DepthwiseConv2dNative':
                    layer['operation'] = 'dwconv'
                layer['activation_type'] = act_type
                layer['input_log2scale'] = ifm_log2scale
                layer['weight_log2scale']  = w_log2scale
                layer['bias_log2scale']  = b_log2scale
                layer['output_log2scale'] = ofm_log2scale
                layer['output_shift'] = ofm_shift
                layer['bias_shift'] = b_shift
                layer['load_bias'] = True
                layer['input_channel_num'] = cin
                layer['output_channel_num'] = cout
                layer['input_size'] = {'height': ifm_size[0], 'width': ifm_size[1]}
                layer['padding'] = {'top': padding_size[0], 'bottom': padding_size[1],
                                    'left': padding_size[2], 'right': padding_size[3]}
                layer['stride'] = {'height': strides[0], 'width': strides[1]}
                layer['kernel_size'] = {'height': kernel_size[0], 'width': kernel_size[1]}
                layer['output_size'] = {'height': ofm_size[0], 'width': ofm_size[1]}
                layer['input_dtype'] = 'int8'
                layer['output_dtype'] = 'int8'
                layer['weight_dtype'] = 'int8'
                layer['bias_dtype'] = 'int8'

                # save the tensors
                np.save(os.path.join(self.output_param_path, layer['name']+'_input.npy'), ifm_quant_data)
                np.save(os.path.join(self.output_param_path, layer['name']+'_weight.npy'), w_quant_data)
                np.save(os.path.join(self.output_param_path, layer['name']+'_bias.npy'), b_quant_data)
                np.save(os.path.join(self.output_param_path, layer['name']+'_output.npy'), ofm_quant_data) 
                                                            
                # assign output_tensor
                output_tensor = ofm_op.outputs[0]

            elif curr_op.type == 'Add':
                # find x/y/ofm ops
                x_op, x_conv_op = self.__get_add_x_op(curr_op)
                y_op, y_conv_op = self.__get_add_y_op(curr_op)
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
                ofm_shift = ofm_log2scale - min(x_log2scale, y_log2scale)

                # calc the quantized ifm/w/b/ofm tensors
                x_quant_data = np.squeeze(self.__quant_tensor(x_data, x_log2scale),0)
                y_quant_data = np.squeeze(self.__quant_tensor(y_data, y_log2scale),0)
                ofm_quant_data = np.squeeze(self.__quant_tensor(ofm_data, ofm_log2scale),0)

                # fill the layer dict
                layer['operation'] = 'add'
                layer['activation_type'] = act_type
                # the 'size' key is deprecated since it relies on in-memory format of tensor
                # layer['size'] = cin*ifm_size[0]*ifm_size[1]
                layer['input_channel_num'] = cin
                layer['input_size'] = {'height': ifm_size[0], 'width': ifm_size[1]}
                layer['dtype'] = 'int8'
                layer['pl_log2scale'] = layer['pl_shiftbit'] = x_log2scale
                layer['add_log2scale'] = layer['add_shiftbit'] = y_log2scale
                layer['output_log2scale'] = ofm_log2scale
                layer['output_shift_bit'] = ofm_shift
                layer['pl_name'] = x_conv_op.name.replace('/', '_')
                layer['add_name'] = y_conv_op.name.replace('/', '_')
                
                # save tensors
                np.save(os.path.join(self.output_param_path, layer['name']+'_pl.npy'), x_quant_data)
                np.save(os.path.join(self.output_param_path, layer['name']+'_add.npy'), y_quant_data)
                np.save(os.path.join(self.output_param_path, layer['name']+'_output.npy'), ofm_quant_data)

                # assign output_tensor
                output_tensor = ofm_op.outputs[0]

            elif curr_op.type == 'Reshape':
                # get reshape params
                # size params (needs to eval tensor)
                shape_tensor = self.__get_reshape_params_from_op(op)
                tensors_to_eval = [op.inputs[0], op.outputs[0], shape_tensor]
                input_data, output_data, shape_data = self.__eval_tensors(tensors_to_eval)
                input_shape = input_data.shape[1:]  # omit the N field
                output_shape = output_data.shape[1:] # omit the N field
                reshape_param = shape_data[1:]    # omit the N field

                # find the nearest previous fakequant, and get the params
                quant_op = self.__get_previous_quant_op(curr_op.inputs[0])
                num_bits, log2scale = self.__get_quant_params_from_op(quant_op)                
                assert num_bits == 8

                # fill the layer dict
                layer['operation'] = 'reshape'
                layer['reshape_param'] = reshape_param.tolist()
                layer['input_shape'] = input_shape
                layer['output_shape'] = output_shape
                layer['dtype'] = 'int8'
                layer['log2scale'] = log2scale

                # get the quantized tensors
                input_quant_data = np.squeeze(self.__quant_tensor(input_data, log2scale),0)
                output_quant_data = np.squeeze(self.__quant_tensor(output_data, log2scale), 0)

                # save tensors 
                np.save(os.path.join(self.output_param_path, layer['name']+'_input.npy'), input_quant_data)
                np.save(os.path.join(self.output_param_path, layer['name']+'_output.npy'), output_quant_data)

                # assign output_tensor
                output_tensor = curr_op.outputs[0]

            elif curr_op.type == 'AvgPool':
                # get ifm/ofm ops
                ifm_op = self.__get_pool_ifm_op(curr_op)
                ofm_op = self.__get_pool_ofm_op(curr_op)

                # conv params
                strides = curr_op.get_attr('strides')[1:3] # (H,W)
                padding = curr_op.get_attr('padding').decode() # 'SAME' or 'VALID'
                kernel_size = curr_op.get_attr('ksize')[1:3] # (H,W)

                # size params (needs to eval tensors)
                # first we evaluate all tensors
                tensors_to_eval = [op.outputs[0] for op in [ifm_op, ofm_op]]
                ifm_data, ofm_data = self.__eval_tensors(tensors_to_eval)
                # then get the sizes
                cin = ifm_data.shape[3]
                ifm_size = ifm_data.shape[1:3] # (H,W)
                cout = ofm_data.shape[3]
                ofm_size = ofm_data.shape[1:3] # (H,W)
                # calculate padding sizes (t,b,l,r)
                padding_size = self.__calc_padding_size(padding, strides, ifm_size, kernel_size, ofm_size)

                # quant_params
                ifm_num_bits, ifm_log2scale = self.__get_quant_params_from_op(ifm_op)
                ofm_num_bits, ofm_log2scale = self.__get_quant_params_from_op(ofm_op)
                input_pre_ls = ofm_log2scale - ifm_log2scale
                assert input_pre_ls >= 0

                # calc the quantized ifm/ofm tensors
                ifm_quant_data = np.squeeze(self.__quant_tensor(ifm_data, ifm_log2scale),0) #HWC
                ofm_quant_data = np.squeeze(self.__quant_tensor(ofm_data, ofm_log2scale),0) # HWC

                # fill layer dict
                layer['operation'] = 'avg_pool'
                layer['input_channel_num'] = cin
                layer['input_size'] = {'height': ifm_size[0], 'width': ifm_size[1]}
                layer['input_log2scale'] = ifm_log2scale
                layer['input_dtype'] = 'int8'
                layer['kernel_size'] = {'height': kernel_size[0], 'width': kernel_size[1]}
                layer['stride'] = {'height': strides[0], 'width': strides[1]}
                layer['padding'] = {'top': padding_size[0], 'bottom': padding_size[1],
                                    'left': padding_size[2], 'right': padding_size[3]}
                layer['output_channel_num'] = cout
                layer['output_size'] = {'height': ofm_size[0], 'width': ofm_size[1]}
                layer['output_log2scale'] = ofm_log2scale
                layer['output_dtype'] = 'int8'
                layer['input_pre_ls'] = input_pre_ls

                # save the tensors
                np.save(os.path.join(self.output_param_path, layer['name']+'_input.npy'), ifm_quant_data)
                np.save(os.path.join(self.output_param_path, layer['name']+'_output.npy'), ofm_quant_data) 
                                                            
                # assign output_tensor
                output_tensor = ofm_op.outputs[0]                 

            else:
                raise ValueError('Unsupported op: ', curr_op)

            # add output tensor to tensors_ready
            tensors_ready.append(output_tensor)
            
            # fill the prev_layers information
            prev_layers = info_prev_layers[curr_op.name]
            layer['previous_layer'] = [s.replace('/','_') for s in prev_layers]

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
                layer['next_layer'] = [s.replace('/', '_') for s in next_layers]
            else:
                layer['next_layer'] = ['endpoint']
                
            # append the current layer to net_def
            self.net_def['layers'].append(layer)

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
        elif op.type == 'AvgPool':
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
        elif ops_found[0].type in VALID_ACTS:
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
        """
        return the output op and the layer op of x
        """
        assert op.type == 'Add'
        op_x = self.__get_real_producer(op.inputs[0])
        assert op_x.type == 'FakeQuantWithMinMaxVars'
        op_layer_x = self.__get_layer_op_from_output(op_x)
        return op_x, op_layer_x
    
    def __get_add_y_op(self, op):
        assert op.type == 'Add'
        op_y = self.__get_real_producer(op.inputs[1])
        assert op_y.type == 'FakeQuantWithMinMaxVars'
        op_layer_y = self.__get_layer_op_from_output(op_y)
        return op_y, op_layer_y

    def __get_layer_op_from_output(self, op):
        """
        The connections after the conv op could be:
        #    CONV   BIAS
        #        \ /
        #        ADD
        #         |
        # (Relu6 or Identities)
        #         |
        #      FakeQuant

        OR

        The connections near the ADD op could be:
        #    IFM0   IFM1
        #        \ /
        #        ADD
        #         |
        # (Relu or Identities)
        #         |
        #      FakeQuant        
        """
        assert op.type == 'FakeQuantWithMinMaxVars'
        op_tmp = self.__get_real_producer(op.inputs[0])
        if op_tmp.type in VALID_ACTS:
            # skip all relu and identities
            op_tmp = self.__get_real_producer(op_tmp.inputs[0])
        
        # now op_tmp should be the add op
        if op_tmp.type != 'Add':
            print(op_tmp)
        assert op_tmp.type == 'Add'
        op_conv = self.__get_real_producer(op_tmp.inputs[0])
        if op_conv.type in VALID_CONV_OPS:
            # the add is bias add
            return op_conv
        elif op_conv.type == 'FakeQuantWithMinMaxVars':
            # this is a add layer
            return op_tmp

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
        elif ops_found[0].type in VALID_ACTS:
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

    def __get_pool_ifm_op(self, op):
        assert(op.type == 'AvgPool')
        op_ifm = self.__get_real_producer(op.inputs[0])
        assert op_ifm.type == 'FakeQuantWithMinMaxVars' or 'Placeholder'
        return op_ifm

    def __get_pool_ofm_op(self, op):
        assert (op.type == 'AvgPool')
        ops_found = self.__get_real_consumers(op.outputs[0])
        assert len(ops_found) == 1
        op_ofm = ops_found[0]
        assert op_ofm.type == 'FakeQuantWithMinMaxVars'
        return op_ofm

    def __get_reshape_params_from_op(self, op):
        """
        return reshape params tensor
        """
        assert op.type == 'Reshape'
        return op.inputs[1]

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
            return 8, 7

    def __get_previous_quant_op(self, tensor):
        """ 
        find the nearest previous fake quant op
        TODO: how to make sure all ops have exactly 1 input?
        """
        while tensor.op.type != 'FakeQuantWithMinMaxVars':
            tensor = tensor.op.inputs[0]
        
        return tensor.op

    def __calc_padding_size(self, padding, strides, ifm_size, kernel_size, ofm_size):
        """
        return: [t, b, l, r]

        How to calculate padding size?
        VALID: all 0
        SAME: 
            pads_along_h = max( (ho-1)*s+k-hi, 0 )
            pads_top = pads_along_h // 2
            pads_bot = pads_along_h - pads_top
            
            Note 1:
            If pads_along_h/w is odd, pads_bot(right) is larger than pads_top(left) by one. 
            Note 2:
            In the above equation the max(x,0) is important when kernel size is less than 
            the stride. In this case, all pads are 0 (do not use negative padding). 
        """
        # calculate padding sizes
        if padding == 'VALID':
            return [0,0,0,0]
        elif padding == 'SAME':
            pads_along_h = max((ofm_size[0]-1)*strides[0]+kernel_size[0]-ifm_size[0],0)
            pads_along_w = max((ofm_size[1]-1)*strides[1]+kernel_size[1]-ifm_size[1],0)
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

    def export_net_def(self):
        """
        export formatted json file
        """
        f = open(self.output_netdef_path, 'w')
        f.write(json.dumps(self.net_def['layers'], indent=2))
        f.close()        

    def test(self):
        g = tf.get_default_graph()
        for op in g.get_operations():
            if op.name == 'BoxPredictor_0/Reshape_1':
                # get reshape params
                # size params (needs to eval tensor)
                shape_tensor = self.__get_reshape_params_from_op(op)
                tensors_to_eval = [op.inputs[0], op.outputs[0], shape_tensor]
                input_data, output_data, shape_data = self.__eval_tensors(tensors_to_eval)
                input_shape = input_data.shape[1:]  # omit the N field
                output_shape = input_data.shape[1:] # omit the N field
                reshape_param = shape_tensor[1:]    # omit the N field

                # size params (needs to eval tensors)
                # first we evaluate all tensors
                # tensors_to_eval = [op.outputs[0] for op in [ifm_op, w_op, b_op, ofm_op]]
                # ifm_data, w_data, b_data, ofm_data = self.__eval_tensors(tensors_to_eval)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract net arch and raw params from quant-aware trained frozen graph')
    parser.add_argument('-i', '--input_path', default='graph.pb', help='input frozen pb')
    parser.add_argument('-o', '--output_path', default='.', help='output directory')
    parser.add_argument('--image_height', type=int, default=160, help='input image height')
    parser.add_argument('--image_width',  type=int, default=160, help='input image width')
    parser.add_argument('--input_tensor_name', default='normalized_input_image_tensor:0', help='input tensor name')
    parser.add_argument('--end_points', default=[
        'BoxPredictor_0/Reshape_1:0', 'BoxPredictor_1/Reshape_1:0', 'BoxPredictor_2/Reshape_1:0', 'BoxPredictor_3/Reshape_1:0', 'BoxPredictor_4/Reshape_1:0', 'BoxPredictor_5/Reshape_1:0', 
        'BoxPredictor_0/Reshape:0', 'BoxPredictor_1/Reshape:0', 'BoxPredictor_2/Reshape:0', 'BoxPredictor_3/Reshape:0', 'BoxPredictor_4/Reshape:0', 'BoxPredictor_5/Reshape:0'], 
        nargs='+', help='final point names')
    parser.add_argument('--test_image', default='test.jpg', help='example input image for evaluating fmaps')
    args = parser.parse_args()

    tf2ir = TF2IR(args.input_path, args.output_path, [args.image_height, args.image_width], args.input_tensor_name, args.end_points, args.test_image)
    tf2ir.load_graph()
    tf2ir.traverse_graph()
    tf2ir.export_net_def()
    # tf2ir.test()

