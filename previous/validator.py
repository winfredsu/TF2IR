import sys, os, shutil
import numpy as np 
from numpy.lib.stride_tricks import as_strided
import json
from munch import *
import PIL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable GPU info
import tensorflow as tf
import time
from subprocess import call

class Validator(object):

    def __init__(self, run_dir='.', graph_path='graph.pb', net_compiled='net_compiled.json', 
        img_path='test.jpg', img_size=160, hw_cstr={'wram':32,'fram':128,'sdram':1536}, param_path='param.npy', 
        mem_map={'param_ost':0, 'buf0_ost':1048576, 'buf1_ost':1310720}, # for mnet 160 0.5, param-1M, buf-256K
        ):
        self.run_dir = run_dir
        self.graph_path = os.path.join(run_dir, graph_path)
        self.net_compiled = os.path.join(run_dir, net_compiled)
        self.img_path = os.path.join(run_dir, img_path)
        self.img_size = img_size
        self.hw_cstr  = hw_cstr
        self.param_path = os.path.join(run_dir, param_path)
        self.mem_map = mem_map

        # initialize the memories
        wram  = np.zeros(hw_cstr['wram']*1024).astype(np.int)
        fram0 = np.zeros(hw_cstr['fram']*1024).astype(np.int)
        fram1 = np.zeros(hw_cstr['fram']*1024).astype(np.int)
        sdram = np.zeros(hw_cstr['sdram']*1024).astype(np.int) # 1.5MB SDRAM    
        self.mem = {'wram': wram, 'fram0': fram0, 'fram1': fram1, 'sdram': sdram}

        self.__prepare_input()
        self.__load_graph()
        self.__load_compiled_net()

    ## Functions for validator initilaization
    def __prepare_input(self):
        self.img = np.array(PIL.Image.open(self.img_path).resize((self.img_size, self.img_size))).astype(np.float)/128-1
        self.img = self.img.reshape(1,self.img_size,self.img_size,3)

    def __load_graph(self):
        self.gd = tf.GraphDef.FromString(open(self.graph_path,'rb').read())

    def __load_compiled_net(self):
        f = open(self.net_compiled)
        self.net = munchify(json.loads(f.read()))
        f.close()

    ## Tensor Utilities 
    def __get_tensors_and_quant(self, tensornames, scale):
        """
        :param tensornames: list of tensornames
        :param scale: list of scale
        """
        if len(tensornames) != len(scale):
            raise ValueError('length of tensornames and scale must be same')
        
        tensors = self.__get_tensors(tensornames)
        return tuple([self.__quant_tensor(tensors[i], scale[i]) for i in range(len(scale))])

    def __get_tensors(self, tensornames):
        """
        :param tensornames: list of tensornames
        return a list of ndarray
        """
        return_elements = ['input:0']+[name+':0' for name in tensornames]
        tensors = tf.import_graph_def(self.gd, return_elements=return_elements)
        inp = tensors[0]
        tensor_values = []
        with tf.Session(graph=inp.graph):
            for tensor in tensors[1:]:
                tensor_values.append(tensor.eval(feed_dict={inp: self.img}))

        return tuple(tensor_values)

    def __quant_tensor(self, tensor, scale):
        return (np.power(2,scale)*tensor).astype(np.int32)

    def __add_rounding_to_bias_and_truncate(self, b, b_ls_raw, ofm_rs):
        """
        add half of OFM LSB to bias, so that the hardware can only perform shifting
        and truncate the bias if the b_ls_raw is negative
        """
        shift = ofm_rs-1-b_ls_raw
        if shift >= 0:
            b += (1<<shift)
        if b_ls_raw < 0:
            b = np.right_shift(b,-b_ls_raw)

        return b
        
    ## Network operations
    def __convdw_slow(self, ifm, w, b, padding, stride, b_ls, ofm_rs):
        """
        ifm: 3d array, HWC
        w: 4d array, HWC1 (the last dim is always 1)
        b: 1d array, C
        padding: [T, L, B, R]
        stride: int
        b_ls: int
        ofm_rs: int

        return: ofm (3d array, HWC)
        """
        hi = ifm.shape[0]
        wi = ifm.shape[1]
        c  = ifm.shape[2]
        k  = w.shape[0]
        
        # 1. insert padding to ifm 
        ifm = self.__insert_padding(ifm, padding)
        hi_padded = ifm.shape[0]
        wi_padded = ifm.shape[1]

        # 2. calculate output size (padding type is VALID)
        ho = int(np.ceil((hi_padded-k+1)/stride))
        wo = int(np.ceil((wi_padded-k+1)/stride))

        # 3. convdw using for loop
        ofm = np.zeros([ho, wo, c]).astype(np.int)
        for c_cnt in range(c):
            ofm_1c = np.zeros([ho, wo])
            for x in range(ho):
                for y in range(wo):
                    for i in range(k):
                        for j in range(k):
                            ofm_1c[x,y] += ifm[x*stride+i][y*stride+j][c_cnt]*w[i][j][c_cnt][0]

            ofm_1c += b[c_cnt]<<b_ls
            ofm[:,:,c_cnt] = ofm_1c

        ofm = np.right_shift(ofm, ofm_rs)
        return self.__relu6(ofm)

    def __convdw(self, ifm, w, b, padding, stride, b_ls, ofm_rs):
        """
        convdw using stride_tricks and einsum

        ifm: 3d array, HWC
        w: 4d array, HWC1 (the last dim is always 1)
        b: 1d array, C
        padding: [T, L, B, R]
        stride: int
        b_ls: int
        ofm_rs: int

        return: ofm (3d array, HWC)
        """
        hi = ifm.shape[0]
        wi = ifm.shape[1]
        c  = ifm.shape[2]
        k  = w.shape[0]
        
        # 1. insert padding to ifm 
        ifm = self.__insert_padding(ifm, padding)
        hi_padded = ifm.shape[0]
        wi_padded = ifm.shape[1]

        # 2. calculate output size (padding type is VALID)
        # ho = int(np.ceil((hi_padded-k+1)/stride))
        # wo = int(np.ceil((wi_padded-k+1)/stride))

        # 3. convdw 
        shape = (k,k,hi_padded-k+1,wi_padded-k+1,c)
        strides = ifm.strides[:-1]+ifm.strides
        ifm_strided = as_strided(ifm,shape,strides)

        # ifm_strided: kkhowoci
        # w: hwcico, co=1
        ofm = np.einsum('ijmn,ijxym->xym',w,ifm_strided)
        ofm = ofm[::stride,::stride,:]+np.left_shift(b,b_ls)

        # ofm = np.zeros([ho, wo, c]).astype(np.int)
        # for c_cnt in range(c):
        #     ofm_1c = np.zeros([ho, wo])
        #     for x in range(ho):
        #         for y in range(wo):
        #             for i in range(k):
        #                 for j in range(k):
        #                     ofm_1c[x,y] += ifm[x*stride+i][y*stride+j][c_cnt]*w[i][j][c_cnt][0]

        #     ofm_1c += b[c_cnt]<<b_ls
        #     ofm[:,:,c_cnt] = ofm_1c

        ofm = np.right_shift(ofm, ofm_rs)
        return self.__relu6(ofm)

    def __conv2d_slow(self, ifm, w, b, padding, stride, b_ls, ofm_rs):
        """
        calculate quantized 2d convolution with stride and padding

        ifm: 3d INT array, HWCi
        w: 4d INT array, HWCiCo
        b: 1d INT array, Co
        padding: [T, L, B, R]
        stride: int
        b_ls: int
        ofm_rs: int

        return: ofm (3d array, HWCo)
        """
        hi = ifm.shape[0]
        wi = ifm.shape[1]
        ci = ifm.shape[2]
        k  = w.shape[0]
        co = w.shape[3]

        # 1. insert padding to ifm (R-B-L-T)
        ifm = self.__insert_padding(ifm, padding)
        hi_padded = ifm.shape[0]
        wi_padded = ifm.shape[1]

        # 2. calculate output size (padding type is VALID now)
        ho = int(np.ceil((hi_padded-k+1)/stride))
        wo = int(np.ceil((wi_padded-k+1)/stride))
        
        # 3. conv2d method 1: use for loop
        ofm = np.zeros([ho, wo, co]).astype(np.int)
        for co_cnt in range(co):
            # calc one output channel (ho*wo)
            ofm_1c = np.zeros([ho, wo]).astype(np.int)
            for ci_cnt in range(ci):
                for x in range(ho):
                    for y in range(wo):
                        for i in range(k):
                            for j in range(k):
                                ofm_1c[x,y] += ifm[x*stride+i][y*stride+j][ci_cnt] * w[i][j][ci_cnt][co_cnt]

            ofm_1c += b[co_cnt]<<b_ls 
            ofm[:,:,co_cnt] = ofm_1c

        # quant & round method:
        # 0. add half of OFM LSB to the bias (for nearest rounding)
        # 1. left shift bias to align b with ifm*w
        # 2. right shift
        ofm = np.right_shift(ofm, ofm_rs)
        return self.__relu6(ofm)
    
    def __conv2d(self, ifm, w, b, padding, stride, b_ls, ofm_rs):
        """
        calculate quantized 2d convolution with stride and padding,
        using as_strided and einsum

        ifm: 3d INT array, HWCi
        w: 4d INT array, HWCiCo
        b: 1d INT array, Co
        padding: [T, L, B, R]
        stride: int
        b_ls: int
        ofm_rs: int

        return: ofm (3d array, HWCo)
        """
        hi = ifm.shape[0]
        wi = ifm.shape[1]
        ci = ifm.shape[2]
        k  = w.shape[0]
        co = w.shape[3]

        # 1. insert padding to ifm (R-B-L-T)
        ifm = self.__insert_padding(ifm, padding)
        hi_padded = ifm.shape[0]
        wi_padded = ifm.shape[1]

        # 2. calculate output size (padding type is VALID now)
        ho = int(np.ceil((hi_padded-k+1)/stride))
        wo = int(np.ceil((wi_padded-k+1)/stride))

        # 3. conv2d method 2: use as_strided and einsum
        # ofm = np.zeros([ho, wo, co]).astype(np.int)

        shape = (k,k,hi_padded-k+1,wi_padded-k+1,ci)
        strides = ifm.strides[:-1] + ifm.strides
        ifm_strided = as_strided(ifm,shape,strides)

        # ifm_strided: k,k,ho,wo,ci
        # weight: k,k,ci,co
        ofm = np.einsum('ijmn,ijxym->xyn',w,ifm_strided)
        ofm = ofm[::stride,::stride,:]+np.left_shift(b,b_ls)

        # quant & round method:
        # 0. add half of OFM LSB to the bias (for nearest rounding)
        # 1. left shift bias to align b with ifm*w
        # 2. right shift
        ofm = np.right_shift(ofm, ofm_rs)
        return self.__relu6(ofm)
    
    def __insert_padding(self, fm, padding):
        """
        padding: [T, L, B, R]
        """
        fm = np.pad(fm, ((padding[0], padding[2]),(padding[1], padding[3]),(0,0)), 'constant')
        return fm

    def __relu6(self, fm):
        fm[fm<0] = 0
        fm[fm>96] = 96
        return fm

    ## HW behavior simulation (hw_block process and mem operations)
    def __process_ce_block(self, op_prop, dump_tensor=False):
        """
        How to process a CE block?
        1. get ifm and w tensors from sram
        2. convolve
        3. save ofm to sram
        """
        conv_type = op_prop.type
        ifm_ocm = op_prop.ifm_ocm
        ofm_ocm = op_prop.ofm_ocm
        # convert base addrs to byte addresses
        ifm_base = op_prop.ifm_base*16
        w_base = op_prop.w_base*4
        b_base = op_prop.b_base*4
        ofm_base = op_prop.ofm_base*16
        # ifm_base = op_prop.ifm_base
        # w_base = op_prop.w_base
        # b_base = op_prop.b_base
        # ofm_base = op_prop.ofm_base
        k = op_prop.k
        stride = op_prop.stride
        padding = op_prop.padding
        hi_size = op_prop.hi_size
        wi_size = op_prop.wi_size
        wi_size_aligned = op_prop.wi_size_aligned
        ci_size = op_prop.ci_size
        ho_size = op_prop.ho_size
        wo_size = op_prop.wo_size
        wo_size_aligned = op_prop.wo_size_aligned
        co_size = op_prop.co_size
        b_ls = op_prop.b_ls
        ofm_rs = op_prop.ofm_rs

        # 1a. we first extract ifm from ifm_ocm
        ifm_shape = [hi_size, wi_size, ci_size]
        ifm = self.__mem2tensor(
            mode='fm', mem=self.mem[ifm_ocm], start=ifm_base, shape=ifm_shape)

        # 1b. then we extract w from wram
        if conv_type == 'conv2d':
            w_shape = [k,k,ci_size,co_size]
            # w_size = self.__calc_mem_size(w_shape,'w_conv2d')
            if stride == 2:
                w = self.__mem2tensor(mode='w_2d_s2', mem=self.mem['wram'], start=w_base, shape=w_shape)
            elif stride == 1:
                w = self.__mem2tensor(mode='w_2d_s1', mem=self.mem['wram'], start=w_base, shape=w_shape)
            else:
                raise ValueError('Unsupported Stride.')            

        elif conv_type == 'convdw':
            w_shape = [k,k,ci_size,1]
            # w_size = self.__calc_mem_size(w_shape,'w_convdw')
            if stride == 2:
                w = self.__mem2tensor(mode='w_dw_s2', mem=self.mem['wram'], start=w_base, shape=w_shape)
            elif stride == 1:
                w = self.__mem2tensor(mode='w_dw_s1', mem=self.mem['wram'], start=w_base, shape=w_shape)
            else:
                raise ValueError('Unsupported Stride.')         
                
        # 1c. then we extract b from wram
        b_shape = [co_size]
        # b_size = self.__calc_mem_size(b_shape, 'b')
        b = self.__mem2tensor(
            mode='b', mem=self.mem['wram'], start=b_base, shape=b_shape)

        # 2. calculate the ofm
        if conv_type == 'conv2d':
            ofm = self.__conv2d(ifm,w,b,padding,stride,b_ls,ofm_rs)
        elif conv_type == 'convdw':
            ofm = self.__convdw(ifm,w,b,padding,stride,b_ls,ofm_rs)

        # 3. finally, save ofm to ofm_ocm
        ofm_len = ho_size*wo_size_aligned*co_size
        self.mem[ofm_ocm][ofm_base:ofm_base+ofm_len]=self.__tensor2mem('fm',ofm)

        if dump_tensor:
            np.save(os.path.join(self.run_dir,'hex_val/ifm.npy'), ifm)
            np.save(os.path.join(self.run_dir,'hex_val/ofm.npy'), ofm)
            np.save(os.path.join(self.run_dir,'hex_val/w.npy'), w)
            np.save(os.path.join(self.run_dir,'hex_val/b.npy'), b)

    def __process_dma_block(self, op_prop):
        # TODO: loop2 may be inccorect!

        # First we need to convert the word addresses to byte addresses
        # Note that lp0_size should be in bytes but lp1_size is a number
        tdir = op_prop.tdir
        tsize = op_prop.tsize*4
        ocm = op_prop.ocm
        ost0 = op_prop.ost0*4
        ost1 = op_prop.ost1*4
        lp0_size = op_prop.lp0_size*4
        if op_prop.lp1_ost == None:
            lp1_ost = 0
        else:
            lp1_ost = op_prop.lp1_ost*4
        if op_prop.lp1_size == None:
            lp1_size = 0
        else:
            lp1_size = op_prop.lp1_size
        if op_prop.lp2_ost == None:
            lp2_ost = 0
        else:
            lp2_ost = op_prop.lp2_ost*4

        tcnt = 0
        lp1_cnt = 0
        lp2_cnt = 0

        if tdir == 1: # sdram to sram
            while tcnt < tsize:
                self.mem[ocm][ost0+tcnt:ost0+tcnt+lp0_size] = \
                    self.mem['sdram'][ost1+lp1_cnt*lp1_ost+lp2_cnt*lp2_ost
                        :ost1+lp1_cnt*lp1_ost+lp2_cnt*lp2_ost+lp0_size]
                lp1_cnt += 1
                tcnt += lp0_size
                if lp1_cnt == lp1_size:
                    lp1_cnt = 0
                    lp2_cnt += 1

        elif tdir == 0: # sram to sdram
            while tcnt < tsize:
                self.mem['sdram'][ost1+lp1_cnt*lp1_ost+lp2_cnt*lp2_ost
                    :ost1+lp1_cnt*lp1_ost+lp2_cnt*lp2_ost+lp0_size] = \
                        self.mem[ocm][ost0+tcnt:ost0+tcnt+lp0_size]
                lp1_cnt += 1
                tcnt += lp0_size
                if lp1_cnt == lp1_size:
                    lp1_cnt = 0
                    lp2_cnt += 1

    def __calc_mem_size(self, shape, mode):
        """
        calculate mem size in byte by tensor shape

        if mode == 'b', shape is in [C]
        if mode == 'fm', shape is in [H,W,C]
        if mode == 'w_conv2d', shape is in [H,W,Ci,Co]
        if mode == 'w_convdw', shape is in [H,W,Ci,Co=1]
        """
        if mode=='b':
            return 2*shape[0]
        elif mode=='fm':
            return int(shape[0]*16*np.ceil(shape[1]/16)*shape[2])
        elif mode=='w_conv2d':
            return shape[0]*shape[1]*shape[2]*shape[3]
        elif mode=='w_convdw':
            return 12*shape[2]

    def __set_sdram(self, start, size, data):
        self.mem['sdram'][start:start+size]=data

    def __tensor2mem(self, mode, tensor):
        """
        convert a tensor to algined memory

        :param mode: 'fm', 'w_2d_s1/2', 'w_dw_s1/2', 'b'
        :param tensor: HWC/HWCiCo/C
        :return mem: 1D array, aligned
        """
        if mode == 'fm':
            # tensor shape: HWC
            # mem shape: CHWaligned
            if len(tensor.shape) != 3:
                raise ValueError("tensor must be a 3D array")
            h = tensor.shape[0]
            w = tensor.shape[1]
            c = tensor.shape[2]
            w_aligned = int(16*np.ceil(w/16))
            mem = np.zeros([c,h,w_aligned]).astype(np.int)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        mem[k,i,j] = tensor[i,j,k]
            mem = mem.reshape(c*h*w_aligned)
            return mem
        elif mode == 'w_2d_s2':
            # tensor shape: HWCiCo
            # mem shape: Co/4-Ci-HW-4Co
            # if stride = 2, the order along W should be 0-2-1
            # and when stride = 2, the kernel size must be 0. 
            if tensor.shape[0]!=3 or tensor.shape[1]!=3:
                raise ValueError('kernel size must be 3x3 for w_2d_s2')
            
            tensor_reordered = np.zeros(tensor.shape).astype(np.int)
            tensor_reordered[:,0,:,:] = tensor[:,0,:,:]
            tensor_reordered[:,1,:,:] = tensor[:,2,:,:]
            tensor_reordered[:,2,:,:] = tensor[:,1,:,:]
            return(self.__tensor2mem('w_2d_s1', tensor_reordered))
        elif mode == 'w_2d_s1':
            # tenosr shape: HWCiCo
            # mem shape: Co/4-Ci-HW-4Co
            h = tensor.shape[0]
            w = tensor.shape[1]
            ci = tensor.shape[2]
            co = tensor.shape[3]
            if co%4 != 0:
                print(tensor.shape)
                raise ValueError("Co must be multiply of 4")
            mem = np.zeros([int(co/4), ci, h, w, 4]).astype(np.int)
            for i in range(int(co/4)):
                for j in range(ci):
                    for k in range(h):
                        for l in range(w):
                            mem[i,j,k,l,:] = tensor[k,l,j,i*4:i*4+4]
            return mem.reshape(h*w*ci*co)
        elif mode == 'w_dw_s2':
            # tensor shape: HWCiCo=1
            # mem shape: Ci-W-H {147x, 258x, 369x}
            if tensor.shape[0]!=3 or tensor.shape[1]!=3:
                raise ValueError('kernel size must be 3x3 for w_dw_s2')
            
            tensor_reordered = np.zeros(tensor.shape).astype(np.int)
            tensor_reordered[:,0,:,:] = tensor[:,0,:,:]
            tensor_reordered[:,1,:,:] = tensor[:,2,:,:]
            tensor_reordered[:,2,:,:] = tensor[:,1,:,:]
            return(self.__tensor2mem('w_dw_s1', tensor_reordered))       
        elif mode == 'w_dw_s1':
            # tensor shape: HWCiCo=1
            # mem shape: Ci-W-H {174x, 285x, 396x}
            h = tensor.shape[0]
            w = tensor.shape[1]
            ci = tensor.shape[2]
            co = tensor.shape[3]
            if h != 3 or w != 3:
                raise ValueError("Kernel size must be 3 for CONVDW")
            if ci%4 != 0:
                print(tensor.shape)
                raise ValueError("Ci must be multiply of 4")
            mem = np.zeros(12*ci).astype(np.int)
            for i in range(ci):
                for j in range(w):
                    mem[12*i+4*j:12*i+4*j+3] = tensor[:,j,i,0]
                    mem[12*i+4*j+3] = 0
            return mem
        elif mode == 'b':
            c = tensor.shape[0]
            mem = np.zeros(2*c).astype(np.int)
            for i in range(c):
                mem[2*i] = np.bitwise_and(tensor[i],255).astype(np.int8).astype(np.int) # LSB
                mem[2*i+1] = np.bitwise_and(np.right_shift(tensor[i],8),255).astype(np.int8).astype(np.int) # MSB
            # print(tensor)
            # print(mem)
            return mem
        else:
            raise ValueError('Unsupported tensor2mem mode')         

    def __mem2tensor(self, mode, mem, start, shape):
        """
        convert a 1d mem array to tensor with given shape

        :param mode: 'fm', 'w_2d_s1/2', 'w_dw_s1/2', 'b'
        :param mem: 1d array containing the tensor
        :param start: start index of the slice
        :param shape: [H,W,C]/[HWCiCo]/[Co]
        :return tensor: tensor with given shape
        """
        if mode == 'fm':
            if len(shape) != 3:
                raise ValueError("shape must be a 3D array")
            h = shape[0]
            w = shape[1]
            c = shape[2]
            w_aligned = int(16*np.ceil(w/16))
            mem_size = self.__calc_mem_size(shape,'fm')
            mem = mem[start:start+mem_size]
            mem = mem.reshape([c,h,w_aligned])
            tensor = np.zeros([h,w,c]).astype(np.int)
            for i in range(h):
                for j in range(w):
                    for k in range(c):
                        tensor[i,j,k] = mem[k,i,j]
            return tensor
        elif mode == 'w_2d_s2':
            # for w_2d_s2, kernel size must be 3x3
            if shape[0]!=3 or shape[1]!=3:
                raise ValueError('kernel size must be 3x3 for w_2d_s2')
            tensor = self.__mem2tensor('w_2d_s1', mem, start, shape)
            tensor_reordered = np.zeros(shape).astype(np.int)
            tensor_reordered[:,0,:,:] = tensor[:,0,:,:]
            tensor_reordered[:,2,:,:] = tensor[:,1,:,:]
            tensor_reordered[:,1,:,:] = tensor[:,2,:,:]
            return tensor_reordered
        elif mode == 'w_2d_s1':
            h = shape[0]
            w = shape[1]
            ci = shape[2]
            co = shape[3]
            if co%4 != 0:
                raise ValueError("Co must be multiply of 4")
            mem_size = self.__calc_mem_size(shape,'w_conv2d')
            mem = mem[start:start+mem_size].reshape([int(co/4), ci, h, w, 4])
            tensor = np.zeros(shape).astype(np.int)
            for i in range(int(co/4)):
                for j in range(ci):
                    for k in range(h):
                        for l in range(w):
                            tensor[k,l,j,i*4:i*4+4] = mem[i,j,k,l,:]
            return tensor
        elif mode == 'w_dw_s2':
            # for w_dw_s2, kernel size must be 3x3
            if shape[0]!=3 or shape[1]!=3:
                raise ValueError('kernel size must be 3x3 for w_dw_s2')
            tensor = self.__mem2tensor('w_dw_s1', mem, start, shape)
            tensor_reordered = np.zeros(shape).astype(np.int)
            tensor_reordered[:,0,:,:] = tensor[:,0,:,:]
            tensor_reordered[:,2,:,:] = tensor[:,1,:,:]
            tensor_reordered[:,1,:,:] = tensor[:,2,:,:]
            return tensor_reordered
        elif mode == 'w_dw_s1':
            h = shape[0]
            w = shape[1]
            ci = shape[2]
            co = shape[3]
            if h != 3 or w != 3:
                raise ValueError("Kernel size must be 3 for CONVDW")
            mem_size = self.__calc_mem_size(shape, 'w_convdw')
            mem = mem[start:start+mem_size]
            tensor = np.zeros(shape).astype(np.int)
            for i in range(ci):
                for j in range(w):
                    tensor[:,j,i,0] = mem[12*i+4*j:12*i+4*j+3]
            return tensor            
        elif mode == 'b':
            mem_size = self.__calc_mem_size(shape,'b')
            mem = mem[start:start+mem_size]
            tensor = np.zeros(shape).astype(np.int)
            for co in range(shape[0]):
                LSB = mem[2*co].astype(np.uint8)
                MSB = mem[2*co+1].astype(np.uint8)
                b = (np.left_shift(MSB,8)+LSB).astype(np.int16)
                tensor[co] = b.astype(np.int)
                # print(tensor)
            return tensor
        else:
            raise ValueError('Unsupported mem2tensor mode')

    def __load_img_to_sdram(self):
        print("####################################")
        print("Loading image to SDRAM img buf0")
        # we scale the img back to [-128,127]
        img = (128*self.img.reshape(self.img_size,self.img_size,3)).astype(np.int)
        self.img_byte_size = self.__calc_mem_size(img.shape,'fm')
        self.__set_sdram(self.mem_map['buf0_ost'], self.__calc_mem_size(img.shape,'fm'), self.__tensor2mem('fm',img))
        print("####################################")

    def __load_params_to_sdram(self):
        print("####################################")
        print("Loading params to SDRAM")
        if os.path.exists(self.param_path):
            param_mem = np.load(self.param_path)
        else:
            param_mem = np.array([])

            for l in self.net.layers:
                w, b = self.__get_tensors_and_quant(
                    tensornames=[l.tensors.w, l.tensors.b],
                    scale=[l.layer_params.w_scale, l.layer_params.b_scale])
                # w = self.__quant_tensor(w, layer.layer_params.w_scale)
                # b = self.__quant_tensor(b, layer.layer_params.b_scale)
                b = self.__add_rounding_to_bias_and_truncate(b, l.layer_params.b_ls_raw, l.layer_params.ofm_rs)
                
                if l.layer_params.op == 'Conv2D':
                    if l.layer_params.stride == 2:
                        w_mem = self.__tensor2mem('w_2d_s2',w)
                    elif l.layer_params.stride == 1:
                        w_mem = self.__tensor2mem('w_2d_s1',w)
                    else:
                        raise ValueError('Unsupported Stride.')

                elif l.layer_params.op == 'DepthwiseConv2dNative':
                    if l.layer_params.stride == 2:
                        w_mem = self.__tensor2mem('w_dw_s2',w)
                    elif l.layer_params.stride == 1:
                        w_mem = self.__tensor2mem('w_dw_s1',w)
                    else:
                        raise ValueError('Unsupported Stride.')

                b_mem = self.__tensor2mem('b',b)
                param_mem = np.concatenate((param_mem,w_mem,b_mem))
            np.save(self.param_path, param_mem)
        print('Parameter size: ',param_mem.shape)
        self.param_conv_byte_size = param_mem.shape[0]
        self.__set_sdram(self.mem_map['param_ost'],len(param_mem),param_mem)
        print("####################################")

    def __init_sdram(self):
        # first load input image to sdram
        self.__load_img_to_sdram()
        # then load model parameters to sdram
        self.__load_params_to_sdram()

    ## RTL simulation utilities
    def __dump_mem_hex(self, mem, hex_dir, hex_prefix, num_rows, num_banks, num_cols, word_width, bit_interleaved=False, mem_row_width=512):
        """
        dump mem to hex files
        the size of 1d memory should be: num_rows*num_banks*num_cols*(word_width/8)
        the num_rows/banks/cols are logical, and mem_row_width is physical width

        :param mem: memory to be dumped
        :param hex_dir: directory of output hex files
        :param hex_prefix: prefix of output hex files
        :param num_rows: number of memory rows
        :param num_banks: number of banks
        :param num_cols: number of memory columns
        :param word_width: bit number of a word (should be same as the SRAM width)
        :param bit_interleaved: to interleave bit along row
        """
        # do param check
        if mem.shape[0] != num_rows*num_banks*num_cols*word_width/8:
            raise ValueError('mem size does not match num_row/bank/col')
        stride_col  = int(word_width/8)     # stride in byte
        stride_bank = stride_col*num_cols
        stride_row  = stride_bank*num_banks
        stride_mem_row = int(mem_row_width/8)
        stride_word = int(word_width/8)

        # reshape the mem to banks
        banks = np.zeros((num_banks, num_rows*num_cols*stride_col)).astype(np.int)
        for i_row in range(num_rows):
            for i_bank in range(num_banks):
                banks[i_bank,i_row*stride_bank:(i_row+1)*stride_bank] = \
                    mem[i_row*stride_row+i_bank*stride_bank:i_row*stride_row+(i_bank+1)*stride_bank]

        # dump the banks to hex (word_width aligned)
        for i_bank in range(num_banks):
            if num_banks==1:
                f = open(os.path.join(hex_dir,hex_prefix+'.hex'),'w')
            else:
                f = open(os.path.join(hex_dir,hex_prefix+'_bank'+str(i_bank)+'.hex'),'w')

            if bit_interleaved==False:
                # for sdram, bit_interleaved=False
                for i_word in range(num_rows*num_cols):
                    word = banks[i_bank,i_word*stride_col:(i_word+1)*stride_col]
                    for i_byte in range(stride_col):
                        # to convert int to hex using 'format', we need to convert it to uint8 first
                        f.write('{:02x}'.format(word[stride_col-1-i_byte].astype(np.uint8)))
                    f.write('\n')
                    # write stride_col INT8 into HEX and '\n'
            else:
                # for sram memory models, bit_interleaved=True
                num_mem_row = int(num_rows*num_cols*word_width/mem_row_width)
                num_word_mem_row = int(mem_row_width/word_width)
                for i_mem_row in range(num_mem_row):
                    mem_row = banks[i_bank, i_mem_row*stride_mem_row:(i_mem_row+1)*stride_mem_row]
                    mem_row_bitstring = []
                    for i_word in range(num_word_mem_row): # how many words in one mem_row
                        word = mem_row[i_word*stride_word:(i_word+1)*stride_word]
                        word_bitstring = ''
                        for i_byte in range(stride_word):
                            word_bitstring += '{:08b}'.format(word[stride_word-1-i_byte].astype(np.uint8))
                        mem_row_bitstring.append(word_bitstring)
                    # do bit interleaving operation
                    mem_row_bitstring_interleaved=['0']*mem_row_width
                    for i in range(word_width):
                        for j in range(num_word_mem_row):
                            mem_row_bitstring_interleaved[i*num_word_mem_row+num_word_mem_row-1-j] = mem_row_bitstring[j][i] # jth word, ith bit
                    for i in range(mem_row_width):
                        f.write(mem_row_bitstring_interleaved[i])
                    f.write('\n')
            f.close()
    
    def __dump_hex_all(self, hex_dir):
        """
        save all memories except for the IRAM
        """
        # SDRAM: param+buf0 = 1.5MB in total, colsize=512(2^9), num_rows=384
        self.__dump_mem_hex(mem=self.mem['sdram'], hex_dir=hex_dir, hex_prefix='sdram', num_rows=int(self.hw_cstr['sdram']*1024/4/512/(16/8)), num_banks=4, num_cols=512, word_width=16)
        # WRAM: 32kB, banksize=1, 32-bit
        self.__dump_mem_hex(mem=self.mem['wram'], hex_dir=hex_dir, hex_prefix='wram', num_rows=int(self.hw_cstr['wram']*1024/1/1/(32/8)), num_banks=1, num_cols=1, word_width=32)
        # FRAM0: 128kB, 32-bit, num_col=1, num_bank=4, num_row=8192
        self.__dump_mem_hex(mem=self.mem['fram0'], hex_dir=hex_dir, hex_prefix='fram0', num_rows=int(self.hw_cstr['fram']*1024/4/1/(32/8)), num_banks=4, num_cols=1, word_width=32)
        # FRAM1: 128kB, 32-bit, num_col=1, num_bank=4, num_row=8192
        self.__dump_mem_hex(mem=self.mem['fram1'], hex_dir=hex_dir, hex_prefix='fram1', num_rows=int(self.hw_cstr['fram']*1024/4/1/(32/8)), num_banks=4, num_cols=1, word_width=32)

    def __load_mem_hex(self, hex_dir, hex_prefix, num_rows, num_banks, num_cols, data_width):
        """
        load hex files exported from Questasim, and combine them into one memory

        :param mem: 1D memory array
        :param hex_prefix: prefix of hex files
        :param num_rows: num of rows
        :param num_banks: num of banks
        :param num_cols: num of columns
        :param data_width: num of bits in a word

        :return mem
        """
        stride_col  = num_bytes = int(data_width/8)
        stride_bank = stride_col*num_cols
        stride_row  = stride_bank*num_banks
        size_col    = num_cols*stride_col
        size_bank   = num_rows*num_cols*stride_col
        size_mem    = num_rows*num_cols*num_banks*stride_col

        mem = np.zeros(size_mem).astype(np.int)

        # read hex files into memory
        for i_bank in range(num_banks):
            # read each file in 
            if num_banks == 1:
                f = open(os.path.join(hex_dir, hex_prefix+'.hex'),'r')
            else:
                f = open(os.path.join(hex_dir, hex_prefix+'_bank'+str(i_bank)+'.hex'),'r')
            
            lines_raw = f.readlines()
            lines = []
            for line in lines_raw:
                if line[0:2]!='//' and line!='\n':
                    lines.append(line)

            idx = 0
            for i_row in range(num_rows):
                for i_col in range(num_cols):
                    # read one line (one data word), convert to integers and write into mem
                    s = lines[idx]
                    idx+=1
                    mem_start = i_row*stride_row+i_bank*stride_bank+i_col*stride_col                    
                    for i_byte in range(num_bytes):
                        hex_str = s[2*(num_bytes-1-i_byte):2*(num_bytes-i_byte)]
                        # here we need to first convert the hex to int and then to int8, or else
                        # negative values will be translated incorrectly
                        hex_uint8 = int(hex_str, 16)
                        if hex_uint8>127:
                            mem[mem_start+i_byte] = hex_uint8-256
                        else:
                            mem[mem_start+i_byte] = hex_uint8
            f.close()
        return mem

    def __load_hex_all(self, hex_dir):
        sdram = self.__load_mem_hex(hex_dir, 'sdram', int(self.hw_cstr['sdram']/4), 4, 512, 16)
        wram  = self.__load_mem_hex(hex_dir, 'wram',  int(self.hw_cstr['wram']*1024/4), 1, 1, 32)
        fram0 = self.__load_mem_hex(hex_dir, 'fram0', int(self.hw_cstr['fram']*1024/16), 4, 1, 32)
        fram1 = self.__load_mem_hex(hex_dir, 'fram1', int(self.hw_cstr['fram']*1024/16), 4, 1, 32)
        return (sdram, wram, fram0, fram1)

    def __run_behavior_sim(self, hwb_end, hwb_start, use_hw_blocks_opt=False, dump_tensor=False):
        """
        run behavior level sim from hwb_start to the hwb before hwb_end
        """
        print('########################')
        if not use_hw_blocks_opt:
            sim_enable = False
            for l in self.net.layers:
                for idx_hwb, hwb in enumerate(l.hw_blocks):
                    if l.id==hwb_start['layer_id'] and idx_hwb==hwb_start['hwb_id']:
                        sim_enable=True
                    if l.id==hwb_end['layer_id'] and idx_hwb==hwb_end['hwb_id']:
                        sim_enable=False
                        print('Ran behavior-level sim from ',hwb_start, 'to ',hwb_end)
                        print('########################')
                        return
                    
                    if sim_enable:
                        if hwb.op_name == 'dma':
                            self.__process_dma_block(hwb.op_prop)
                        elif hwb.op_name == 'ce':
                            self.__process_ce_block(hwb.op_prop, dump_tensor)
        else:
            for hwb in self.net.hw_blocks_opt:
                if hwb.op_name=='dma':
                    self.__process_dma_block(hwb.op_prop)
                elif hwb.op_name=='ce':
                    self.__process_ce_block(hwb.op_prop, dump_tensor)

    def __dump_inst(self, hex_dir, hwb_start, hwb_end, append_int, use_hw_blocks_opt=False):
        """
        dump instructions from hwb start to the hwb before hwb_end
        
        :param hex_dir: output dir
        :param hwb_start: {layer_id: 'Conv2d_0', hwb_id: 0}
        :param append_int: whether add a 'disable_fetch & int' instruction at the end
        :param use_hw_blocks_opt: whether to use optimized hw blocks
        """
        if not use_hw_blocks_opt:
            f = open(os.path.join(hex_dir, 'iram.hex'), 'w')
            dump_enable = False
            for l in self.net.layers:
                for idx, hwb in enumerate(l.hw_blocks):
                    if l.id == hwb_start['layer_id'] and idx == hwb_start['hwb_id']:
                        dump_enable = True
                    if l.id == hwb_end['layer_id'] and idx == hwb_end['hwb_id']:
                        dump_enable = False
                        if append_int:
                            f.write('{:08x}'.format((3<<24)+(1<<16)+(1<<17)+(1<<1)))
                            f.write('\n')
                        f.close()
                        return
                    if dump_enable:
                        for line in hwb.binaries:
                            f.write(line)
                            f.write('\n')
            if append_int:
                f.write('{:08x}'.format((3<<24)+(1<<16)+(1<<17)+(1<<1)))
                f.write('\n')
            f.close()

        else:
            f = open(os.path.join(hex_dir, 'iram.hex'), 'w')
            for hwb in self.net.hw_blocks_opt:
                for line in hwb.binaries:
                    f.write(line)
                    f.write('\n')

            if append_int:
                f.write('{:08x}'.format((3<<24)+(1<<16)+(1<<17)+(1<<1)))
                f.write('\n')
            f.close()

    def __run_rtl_sim(self, hex_py2rtl_path, hex_rtl2py_path):

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(self.run_dir)))
        sim_dir = os.path.join(root_dir, 'sim/netlist')
        f = open(os.path.join(sim_dir, 'ram_path.sv'),'w')
        f.write('`define IRAM_IN_PATH "'+os.path.join(hex_py2rtl_path, 'iram.hex').replace('\\', '/')+'"\n')        
        f.write('`define WRAM_IN_PATH "'+os.path.join(hex_py2rtl_path, 'wram.hex').replace('\\', '/')+'"\n')        
        f.write('`define FRAM0_BANK0_IN_PATH "'+os.path.join(hex_py2rtl_path, 'fram0_bank0.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM0_BANK1_IN_PATH "'+os.path.join(hex_py2rtl_path, 'fram0_bank1.hex').replace('\\', '/')+'"\n')        
        f.write('`define FRAM0_BANK2_IN_PATH "'+os.path.join(hex_py2rtl_path, 'fram0_bank2.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM0_BANK3_IN_PATH "'+os.path.join(hex_py2rtl_path, 'fram0_bank3.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM1_BANK0_IN_PATH "'+os.path.join(hex_py2rtl_path, 'fram1_bank0.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM1_BANK1_IN_PATH "'+os.path.join(hex_py2rtl_path, 'fram1_bank1.hex').replace('\\', '/')+'"\n')        
        f.write('`define FRAM1_BANK2_IN_PATH "'+os.path.join(hex_py2rtl_path, 'fram1_bank2.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM1_BANK3_IN_PATH "'+os.path.join(hex_py2rtl_path, 'fram1_bank3.hex').replace('\\', '/')+'"\n')
        f.write('`define SDRAM_BANK0_IN_PATH "'+os.path.join(hex_py2rtl_path, 'sdram_bank0.hex').replace('\\', '/')+'"\n')
        f.write('`define SDRAM_BANK1_IN_PATH "'+os.path.join(hex_py2rtl_path, 'sdram_bank1.hex').replace('\\', '/')+'"\n')        
        f.write('`define SDRAM_BANK2_IN_PATH "'+os.path.join(hex_py2rtl_path, 'sdram_bank2.hex').replace('\\', '/')+'"\n')
        f.write('`define SDRAM_BANK3_IN_PATH "'+os.path.join(hex_py2rtl_path, 'sdram_bank3.hex').replace('\\', '/')+'"\n')
        f.write('`define WRAM_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'wram.hex').replace('\\', '/')+'"\n')        
        f.write('`define FRAM0_BANK0_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'fram0_bank0.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM0_BANK1_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'fram0_bank1.hex').replace('\\', '/')+'"\n')        
        f.write('`define FRAM0_BANK2_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'fram0_bank2.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM0_BANK3_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'fram0_bank3.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM1_BANK0_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'fram1_bank0.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM1_BANK1_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'fram1_bank1.hex').replace('\\', '/')+'"\n')        
        f.write('`define FRAM1_BANK2_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'fram1_bank2.hex').replace('\\', '/')+'"\n')
        f.write('`define FRAM1_BANK3_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'fram1_bank3.hex').replace('\\', '/')+'"\n')
        f.write('`define SDRAM_BANK0_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'sdram_bank0.hex').replace('\\', '/')+'"\n')
        f.write('`define SDRAM_BANK1_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'sdram_bank1.hex').replace('\\', '/')+'"\n')        
        f.write('`define SDRAM_BANK2_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'sdram_bank2.hex').replace('\\', '/')+'"\n')
        f.write('`define SDRAM_BANK3_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'sdram_bank3.hex').replace('\\', '/')+'"\n')
        f.write('`define FINAL_RESULT_OUT_PATH "'+os.path.join(hex_rtl2py_path, 'final_result.hex').replace('\\', '/')+'"\n')
        f.write('`define SDRAM_USED_SIZE '+str(int(self.hw_cstr['sdram']*1024/8))+'\n') # words per sdram bank
        f.close()
        print('########################')
        call(['make', 'sim', '-C', sim_dir])
        print('Ran rtl simulation.')
        print('########################')

    ## MCU implementation utilities
    def __dump_mem_header(self, mem, dtype, header_dir, header_prefix, start, size, flash_base_addr): 
        """
        dump mem (1d array) to array in .h
        """
        f = open(os.path.join(header_dir, header_prefix+'.h'), 'w')
        if dtype=='int8':
            f.write('#define '+header_prefix.upper()+'_SIZE '+str(int(size/4))+'\n')
            if flash_base_addr==None:
                f.write('static const uint32_t '+header_prefix+' ['+str(int(size/4))+'] = {')
            else:
                f.write('static const uint32_t '+header_prefix+' ['+str(int(size/4))+'] __attribute__((at('+flash_base_addr+'))) = {')
            for i in range(int(size/4)):
                f.write('0x')
                f.write('{:02x}'.format(mem[start+4*i+3].astype(np.uint8)))
                f.write('{:02x}'.format(mem[start+4*i+2].astype(np.uint8)))
                f.write('{:02x}'.format(mem[start+4*i+1].astype(np.uint8)))
                f.write('{:02x}'.format(mem[start+4*i+0].astype(np.uint8)))
                f.write(',')

        elif dtype=='float32':
            f.write('#define '+header_prefix.upper()+'_SIZE '+str(int(size))+'\n')
            if flash_base_addr==None:
                f.write('static const float32_t '+header_prefix+' ['+str(int(size))+'] = {')
            else:
                f.write('static const float32_t '+header_prefix+' ['+str(int(size))+'] __attribute__((at('+flash_base_addr+'))) = {')

            for i in range(int(size)):
                f.write(str(mem[start+i])+',')

        f.write('};\n')
        f.close()

    def __dump_inst_header(self, header_dir, hwb_start, hwb_end, flash_base_addr, append_int, use_hw_blocks_opt=False):
        """
        dump instructions from hwb start to the hwb before hwb_end
        
        :param hex_dir: output dir
        :param hwb_start: {layer_id: 'Conv2d_0', hwb_id: 0}
        :param append_int: whether add a 'disable_fetch & int' instruction at the end
        :param use_hw_blocks_opt: whether to use optimized hw blocks
        """
        inst = []
        if not use_hw_blocks_opt:
            dump_enable = False
            for l in self.net.layers:
                for idx, hwb in enumerate(l.hw_blocks):
                    if l.id == hwb_start['layer_id'] and idx == hwb_start['hwb_id']:
                        dump_enable = True
                    if l.id == hwb_end['layer_id'] and idx == hwb_end['hwb_id']:
                        dump_enable = False
                        if append_int:
                            inst.append('0x'+'{:08x}'.format((3<<24)+(1<<16)+(1<<17)+(1<<1)))
                        break
                    if dump_enable:
                        for line in hwb.binaries:
                            inst.append('0x'+line)
                if dump_enable == False:
                    break
            if dump_enable and append_int:
                inst.append('0x'+'{:08x}'.format((3<<24)+(1<<16)+(1<<17)+(1<<1)))
        else:
            for hwb in self.net.hw_blocks_opt:
                for line in hwb.binaries:
                    inst.append('0x'+line)

            if append_int:
                inst.append('0x'+'{:08x}'.format((3<<24)+(1<<16)+(1<<17)+(1<<1)))

        f = open(os.path.join(header_dir, 'inst.h'), 'w')
        f.write('#define INST_SIZE '+str(len(inst))+'\n')
        f.write('static const uint32_t inst['+str(len(inst))+'] __attribute__((at('+flash_base_addr+'))) = {')
        for line in inst:
            f.write(line)
            f.write(', ')
        f.write('};\n')
        f.close()

    def __prepare_fc_params(self):
        # 1. get the tensors after the last conv layer
        # ifm: [-8,8), w: [-1,1), b: float
        w, b= self.__get_tensors([
            'MobilenetV1/Logits/Conv2d_1c_1x1/weights_quant/FakeQuantWithMinMaxVars',
            'MobilenetV1/Logits/Conv2d_1c_1x1/biases/read'])
        w = np.squeeze(w, axis=(0,1))
        
        # 2. get the INT8 tensors
        w = self.__quant_tensor(w, 7) 
        # 512x1001, column first, to 1001x512 
        w = w.flatten('F')

        # 3. scale bias to align with ifm and w
        b = 2048*b

        return w,b

    ## Main RTL  Validations
    def validate_at_layer_level(self):
        start = time.clock()
        validation_pass = True
        print('########################')
        print('Validate at layer level:')
        for layer in self.net.layers:
            if self.validate_layer_at_layer_level(layer) == False:
                validation_pass = False
                break
        print('########################')
        if validation_pass:
            print('Validation pass.')
            print('Time used:', time.clock()-start)
        else:
            print('Validation failed.')
        print('########################')

    def validate_layer_at_layer_level(self, layer):
        # 1. get tensors
        ifm, w, b, ofm = self.__get_tensors([layer.tensors.ifm, layer.tensors.w, layer.tensors.b, layer.tensors.ofm])
        ifm = np.squeeze(ifm, axis=0) # remove the N axis
        ofm = np.squeeze(ofm, axis=0) # remove the N axis

        # 2. generate INT tensors and prepare bias
        ifm = self.__quant_tensor(ifm, layer.layer_params.ifm_scale)
        w   = self.__quant_tensor(w, layer.layer_params.w_scale)
        b   = self.__quant_tensor(b, layer.layer_params.b_scale)
        ofm = self.__quant_tensor(ofm, layer.layer_params.ofm_scale)

        b = self.__add_rounding_to_bias_and_truncate(b, layer.layer_params.b_ls_raw, layer.layer_params.ofm_rs)

        padding = layer.layer_params.padding_size
        stride = layer.layer_params.stride
        # ofm_scale = layer.layer_params.ofm_scale

        if layer.layer_params.op == 'Conv2D':
            ofm_ut = self.__conv2d(ifm,w,b,padding,stride,layer.layer_params.b_ls,layer.layer_params.ofm_rs)
        elif layer.layer_params.op == 'DepthwiseConv2dNative':
            ofm_ut = self.__convdw(ifm,w,b,padding,stride,layer.layer_params.b_ls,layer.layer_params.ofm_rs)

        abs_err = np.abs(np.subtract(ofm_ut, ofm))
        
        if np.max(abs_err) == 0.0:
            print('Layer '+layer.id+' validation passed.')
            return True
        else:
            print('Error: Layer '+layer.id+' validation failed.')
            return False

    def validate_at_tile_level(self):
        start = time.clock()

        validation_pass = True
        print('########################')
        print('Validate at tile level:')
        for layer in self.net.layers:
            if self.validate_layer_at_tile_level(layer, self.hw_cstr) == False:
                validation_pass = False
                break
        print('########################')
        if validation_pass:
            print('Validation pass.')
            print("Time used:",time.clock()-start)
        else:
            print('Validation failed.')
        print('########################')

    def validate_layer_at_tile_level(self, layer, hw_cstr):
        """
        validate the tiling params will never cause memory overflow
        and validate the given tiles can produce correct ofm
        """
        hi = layer.layer_params.hi
        wi = layer.layer_params.wi
        ci = layer.layer_params.ci
        ho = layer.layer_params.ho
        wo = layer.layer_params.wo
        co = layer.layer_params.co
        k  = layer.layer_params.k
        stride = layer.layer_params.stride
        b_ls = layer.layer_params.b_ls
        ofm_rs = layer.layer_params.ofm_rs

        wi_t_aligned = layer.tiling_params.wi_t_aligned
        wo_t_aligned = layer.tiling_params.wo_t_aligned

        max_w_size = hw_cstr['wram']*1024/2
        max_fm_size = hw_cstr['fram']*1024/2

        ifm, w, b, ofm = self.__get_tensors([layer.tensors.ifm, layer.tensors.w, layer.tensors.b, layer.tensors.ofm])
        ifm = np.squeeze(ifm, axis=0) # remove the N axis
        ofm = np.squeeze(ofm, axis=0) # remove the N axis

        # quant the tensors and add rounding to b
        ifm = self.__quant_tensor(ifm, layer.layer_params.ifm_scale)
        w   = self.__quant_tensor(w, layer.layer_params.w_scale)
        b   = self.__quant_tensor(b, layer.layer_params.b_scale)
        ofm = self.__quant_tensor(ofm, layer.layer_params.ofm_scale)

        b = self.__add_rounding_to_bias_and_truncate(b, layer.layer_params.b_ls_raw, layer.layer_params.ofm_rs)

        # padding = layer.layer_params.padding_size
        # ofm_scale = layer.layer_params.ofm_scale

        tiles = layer.tiles
        ofm_ut = np.zeros([ho, wo, co]).astype(np.int)
        
        if layer.layer_params.op == 'Conv2D':
            if layer.tiling_params.type == 'SIMPLE':
                for subtiles in tiles:
                    for tile in subtiles:
                        # wram: all w/b of the entire layer
                        # fram0: hi_tile_size rows of all input channels
                        # fram1: ho_tile_size rows of all output channels
                        if co*(k*k*ci+2) > max_w_size:
                            raise ValueError('Error: wram size violation. '+layer.id)
                        if ci*wi_t_aligned*tile.hi_tile_size > max_fm_size:
                            raise ValueError('Error: ifm size violation. '+layer.id)
                        if co*tile.ho_tile_size*wo_t_aligned > max_fm_size:
                            raise ValueError('Error: ofm size violation. '+layer.id)

                        # then we calculate the ofm of this tile and append it to ofm_ut
                        hi_start = tile.hi_tile_start
                        hi_end   = hi_start + tile.hi_tile_size
                        ho_start = tile.ho_tile_start
                        ho_end   = ho_start + tile.ho_tile_size
                        padding = tile.padding

                        ofm_ut[ho_start:ho_end,:,:] = \
                            self.__conv2d(
                                ifm[hi_start:hi_end,:,:],
                                w,
                                b,
                                padding,
                                stride,
                                b_ls,
                                ofm_rs)

            elif layer.tiling_params.type == 'MULTI_OFM':
                for subtiles in tiles:
                    for tile in subtiles:
                        # wram: all w/b of the entire layer
                        # fram0: hi_tile_size rows of all input channels
                        # fram1: ho_tile_size rows of co_tile_size output channels
                        if co*(k*k*ci+2) > max_w_size:
                            raise ValueError('Error: wram size violation. '+layer.id)
                        if ci*wi_t_aligned*tile.hi_tile_size > max_fm_size:
                            raise ValueError('Error: ifm size violation. '+layer.id)
                        if tile.co_tile_size*tile.ho_tile_size*wo_t_aligned > max_fm_size:
                            raise ValueError('Error: ofm size violation, '+layer.id)

                        # then we calculate the ofm of this tile and append it to ofm_ut
                        hi_start = tile.hi_tile_start
                        hi_end   = hi_start + tile.hi_tile_size
                        ho_start = tile.ho_tile_start
                        ho_end   = ho_start + tile.ho_tile_size
                        co_start = tile.co_tile_start
                        co_end   = co_start + tile.co_tile_size
                        padding = tile.padding

                        ofm_ut[ho_start:ho_end,:,co_start:co_end] = \
                            self.__conv2d(
                                ifm[hi_start:hi_end,:,:],
                                w[:,:,:,co_start:co_end],
                                b[co_start:co_end],
                                padding,
                                stride,
                                b_ls,
                                ofm_rs)

            elif layer.tiling_params.type == 'MULTI_W':
                for subtiles in tiles:
                    for tile in subtiles:
                        # wram: w/b of co_tile_size output channels
                        # fram0: hi_tile_size rows of all input channels
                        # fram1: ho_tile_size rows of all output channels
                        if tile.co_tile_size*(k*k*ci+2) > max_w_size:
                            raise ValueError('Error: wram size violation. '+layer.id)
                        if ci*wi_t_aligned*tile.hi_tile_size > max_fm_size:
                            raise ValueError('Error: ifm size violation. '+layer.id)
                        if co*tile.ho_tile_size*wo_t_aligned > max_fm_size:
                            raise ValueError('Error: ofm size violation. '+layer.id)

                        # then we calculate the ofm of this tile and append it to ofm_ut
                        hi_start = tile.hi_tile_start
                        hi_end   = hi_start + tile.hi_tile_size
                        ho_start = tile.ho_tile_start
                        ho_end   = ho_start + tile.ho_tile_size
                        co_start = tile.co_tile_start
                        co_end   = co_start + tile.co_tile_size
                        padding = tile.padding

                        ofm_ut[ho_start:ho_end,:,co_start:co_end] = \
                            self.__conv2d(
                                ifm[hi_start:hi_end,:,:],
                                w[:,:,:,co_start:co_end],
                                b[co_start:co_end],
                                padding,
                                stride,
                                b_ls,
                                ofm_rs)

            elif layer.tiling_params.type == 'MULTI_W_OFM':
                for subtiles in tiles:
                    for tile in subtiles:
                        # wram: w/b of co_tile_size output channels
                        # fram0: hi_tile_size rows of all input channels
                        # fram1: ho_tile_size rows of co_tile_size output channels
                        if tile.co_tile_size*(k*k*ci+2) > max_w_size:
                            raise ValueError('Error: wram size violation. '+layer.id)
                        if ci*wi_t_aligned*tile.hi_tile_size > max_fm_size:
                            raise ValueError('Error: ifm size violation. '+layer.id)
                        if tile.co_tile_size*tile.ho_tile_size*wo_t_aligned > max_fm_size:
                            raise ValueError('Error: ofm size violation. '+layer.id)

                        # then we calculate the ofm of this tile and append it to ofm_ut
                        hi_start = tile.hi_tile_start
                        hi_end   = hi_start + tile.hi_tile_size
                        ho_start = tile.ho_tile_start
                        ho_end   = ho_start + tile.ho_tile_size
                        co_start = tile.co_tile_start
                        co_end   = co_start + tile.co_tile_size
                        padding = tile.padding

                        ofm_ut[ho_start:ho_end,:,co_start:co_end] = \
                            self.__conv2d(
                                ifm[hi_start:hi_end,:,:],
                                w[:,:,:,co_start:co_end],
                                b[co_start:co_end],
                                padding,
                                stride,
                                b_ls,
                                ofm_rs)

            else:
                raise ValueError('Temporarily unsupported tiling type.')

        elif layer.layer_params.op == 'DepthwiseConv2dNative':
            if layer.tiling_params.type in ['SIMPLE', 'MULTI_IFM']:
                for subtiles in tiles:

                    tile = subtiles[0]
                    # first check the weights of the layer, the ifms of the tile and ofms of the tile 
                    # never exceed memory limits
                    if co*(k*k+2) > max_w_size:
                        raise ValueError('Error: wram size violation. '+layer.id)
                    if hi*wi_t_aligned*tile.ci_tile_size > max_fm_size:
                        raise ValueError('Error: ifm size violation. '+layer.id)
                    if ho*wo_t_aligned*tile.ci_tile_size > max_fm_size:
                        raise ValueError('Error: ofm size violation. '+layer.id)
                    
                    # then calculate the ofm of this tile and append it to ofm_ut
                    c_start = tile.ci_tile_start
                    c_end = tile.ci_tile_start + tile.ci_tile_size
                    padding = tile.padding

                    ofm_ut[:,:,c_start:c_end] = \
                        self.__convdw(ifm[:,:,c_start:c_end],
                            w[:,:,c_start:c_end,:],
                            b[c_start:c_end],
                            padding,
                            stride,
                            b_ls,
                            ofm_rs)
            else:
                raise ValueError("Temporarily unsupported tiling type.")

        abs_err = np.abs(np.subtract(ofm_ut, ofm))

        if np.max(abs_err) == 0.0:
            print('Layer '+layer.id+' validation passed.')
            return True
        else:
            print('Error: Layer '+layer.id+' validation failed.')
            return False
    
    def validate_params_in_sdram(self):
        """
        Validate the params in sdram is consistent with tf tensors
        """
        self.__init_sdram()
        print('####################################')
        print('Validating params in SDRAM')
        for layer in self.net.layers:
            w, b = self.__get_tensors([layer.tensors.w, layer.tensors.b])
            w = self.__quant_tensor(w, layer.layer_params.w_scale)
            b = self.__quant_tensor(b, layer.layer_params.b_scale)
            b = self.__add_rounding_to_bias_and_truncate(b, layer.layer_params.b_ls_raw, layer.layer_params.ofm_rs)

            if layer.layer_params.op == 'Conv2D':
                w_ut = self.__mem2tensor_w_conv2d(self.mem['sdram'][
                    layer.mem_info.w_ost:layer.mem_info.w_ost+layer.mem_info.w_size], w.shape)
            elif layer.layer_params.op == 'DepthwiseConv2dNative':
                w_ut = self.__mem2tensor_w_convdw(self.mem['sdram'][
                    layer.mem_info.w_ost:layer.mem_info.w_ost+layer.mem_info.w_size], w.shape)

            abs_err = np.abs(np.subtract(w_ut, w))
            if np.max(abs_err) == 0.0:
                print('Layer '+layer.id+' weight validation passed.')
            else:
                print('Error: Layer '+layer.id+' weight validation failed.')
                return False

            b_ut = self.__mem2tensor_b(self.mem['sdram'][layer.mem_info.b_ost:layer.mem_info.b_ost+layer.mem_info.b_size])
            abs_err = np.abs(np.subtract(b_ut, b))
            if np.max(abs_err) == 0.0:
                print('Layer '+layer.id+' bias validation passed.')
            else:
                print('Error: Layer '+layer.id+' bias validation failed.')
                return False

    def validate_at_hw_block_level(self, use_opt_hw_blocks=False):
        start = time.clock()
        validation_pass = True
        print('########################')
        print('Validate at hw block level:')
        
        # 1. init memory
        self.__init_sdram()

        # 2. iterate over the layers
        dma_cnt = ce_cnt = 0
        if not use_opt_hw_blocks:
            for idx,l in enumerate(self.net.layers):
                # process the hw blocks
                for hlb in l.hw_blocks:
                    if hlb.op_name == 'dma':
                        self.__process_dma_block(hlb.op_prop)
                        dma_cnt += 1
                    elif hlb.op_name == 'ce':
                        self.__process_ce_block(hlb.op_prop)
                        ce_cnt += 1

                # get ofm from TF
                ofm = self.__get_tensors_and_quant([l.tensors.ofm],[l.layer_params.ofm_scale])
                ofm = np.squeeze(ofm, axis=0)
                
                # get ofm_ut from sdram buffer
                ofm_ut_shape = [l.layer_params.ho,l.layer_params.wo,l.layer_params.co]

                ofm_ut = self.__mem2tensor(
                    mode='fm', mem=self.mem['sdram'], start=self.mem_map['buf'+str((idx+1)%2)+'_ost'],
                    shape=ofm_ut_shape)

                validation_pass = (ofm_ut==ofm).all()
                if validation_pass:
                    print('Layer '+l.id+' validation passed.')
                else:
                    print('Layer '+l.id+' validation failed.')
                    print(np.max(np.abs(ofm_ut-ofm)))
                    print(np.sum(ofm_ut!=ofm))
                    return

        else:
            # process the opt_hw_blocks
            for hlb in self.net.hw_blocks_opt:
                if hlb.op_name == 'dma':
                    self.__process_dma_block(hlb.op_prop)
                    dma_cnt += 1
                elif hlb.op_name == 'ce':
                    self.__process_ce_block(hlb.op_prop)
                    ce_cnt += 1

            # check the final result
            l = self.net.layers[-1]
            # get ofm from tf
            ofm = self.__get_tensors_and_quant([l.tensors.ofm],[l.layer_params.ofm_scale])
            ofm = np.squeeze(ofm, axis=0)
            # get ofm_ut from sdram
            ofm_ut_shape = [l.layer_params.ho,l.layer_params.wo,l.layer_params.co]
            ofm_ut = self.__mem2tensor(
                mode='fm', mem=self.mem['sdram'], start=self.mem_map['buf'+str((len(self.net.layers))%2)+'_ost'],
                shape=ofm_ut_shape)
            validation_pass = (ofm_ut==ofm).all()

        print('########################')
        if validation_pass:
            print('Validation passed!')
        else:
            print('Validation failed.')
        print("Time used:",time.clock()-start)
        print('DMA hw_block count = '+str(dma_cnt))
        print('CE  hw_block count = '+str(ce_cnt))
        print('########################')

    def validate_rtl_opt(self):
        """
        Validate RTL using optimized instructions.
        Only the final resutl in SDRAM is checked. 
        """
        hex_py2rtl_path = os.path.abspath(os.path.join(self.run_dir, 'hex_py2rtl'))
        if os.path.exists(hex_py2rtl_path):
            shutil.rmtree(hex_py2rtl_path)    
        os.mkdir(hex_py2rtl_path)
        
        hex_rtl2py_path = os.path.abspath(os.path.join(self.run_dir, 'hex_rtl2py'))
        if os.path.exists(hex_rtl2py_path):
            shutil.rmtree(hex_rtl2py_path)
        os.mkdir(hex_rtl2py_path)

        hex_val_path = os.path.abspath(os.path.join(self.run_dir, 'hex_val'))
        if os.path.exists(hex_val_path):
            shutil.rmtree(hex_val_path)    
        os.mkdir(hex_val_path)

        # 0. init sdram
        self.__init_sdram()

        # 1. dump mem to hex (only SDRAM will be used)
        self.__dump_hex_all(hex_py2rtl_path)
        
        # 2. generate inst from optimized hwbs
        self.__dump_inst(hex_py2rtl_path, hwb_start=None, hwb_end=None, append_int=True, use_hw_blocks_opt=True)
        
        # 4. run behavior level sim from optimized hwbs
        self.__run_behavior_sim(hwb_start=None, hwb_end=None, use_hw_blocks_opt=True)

        # 5. run rtl sim with above instructions (and generate a 'ram_path.sv')
        self.__run_rtl_sim(hex_py2rtl_path, hex_rtl2py_path)

        # 6. load hex generated by RTL sim
        sdram_rtlsim = self.__load_mem_hex(hex_rtl2py_path, 'sdram', int(self.hw_cstr['sdram']/4), 4, 512, 16)
        final_result_rtlsim = self.__load_mem_hex(hex_rtl2py_path, 'final_result', 1024, 1, 1, 32)
        
        # 7. check if sdram is consistent
        print('########################')
        if (self.mem['sdram']==sdram_rtlsim).all() == False:
            print('Validation failed. SDRAM inconsistency detected.')
            self.__dump_mem_hex(self.mem['sdram'], hex_val_path,'sdram', int(self.hw_cstr['sdram']/4), 4, 512, 16)
            return
       
        if (self.mem['sdram'][self.mem_map['buf1_ost']:self.mem_map['buf1_ost']+1024*4]==final_result_rtlsim).all() == False:
            print('Validation failed. FINAL RESULT inconsistency detected.')
            return
        
        print('Validation passed!')
        print('########################')

    def validate_rtl(self, hwb_end, hwb_start={'layer_id': 'Conv2d_0', 'hwb_id': 0}, target='SIM'):
        """
        This validation method run both behavior sim and RTL sim from start
        to a user-defined checkpoint. 
        
        To init SRAMs using $readmemh, TARGET_SIM must be defined in makefile to use the
        generic sram models instead of process-specific sram model. 

        For RTL sim, the RAMs and SDRAMs are first initialized and dumped into hex files.
        The instructions from beginning to the checkpoint is also dumped to hex file
        to initialize iram. A PC instruction is appended at the end of the inst stream 
        to stop fetching and generate a interrupt. And the testbench saves the memories
        after simulation to output hex files. 
        
        To make the comparison of hex files easier, the SRAMs are initialized to 0 for
        RTL sim, which isn't really necessary on silicons. 

        :param hwb_start: start validation from which hw block {'layer_id': 'Conv2d_0', 'hwb_id': 0}
        :param hwb_end: end hw block
        """
        hex_py2rtl_path = os.path.abspath(os.path.join(self.run_dir, 'hex_py2rtl'))
        if os.path.exists(hex_py2rtl_path):
            shutil.rmtree(hex_py2rtl_path)    
        os.mkdir(hex_py2rtl_path)
        
        hex_rtl2py_path = os.path.abspath(os.path.join(self.run_dir, 'hex_rtl2py'))
        if os.path.exists(hex_rtl2py_path):
            shutil.rmtree(hex_rtl2py_path)
        os.mkdir(hex_rtl2py_path)

        hex_val_path = os.path.abspath(os.path.join(self.run_dir, 'hex_val'))
        if os.path.exists(hex_val_path):
            shutil.rmtree(hex_val_path)    
        os.mkdir(hex_val_path)

        # 0. init sdram
        self.__init_sdram()

        # 1. run behavior level sim from initial to hwb_start
        self.__run_behavior_sim(hwb_start={'layer_id': 'Conv2d_0', 'hwb_id': 0}, hwb_end=hwb_start)

        # 2. dump mem to hex
        self.__dump_hex_all(hex_py2rtl_path)
        
        # 3. generate inst from hwb_start to hwb_end
        self.__dump_inst(hex_py2rtl_path, hwb_start, hwb_end, append_int=True)
        
        # 4. run behavior level sim from hw_start to hwb_end
        self.__run_behavior_sim(hwb_start=hwb_start, hwb_end=hwb_end, dump_tensor=True)

        # 5. run rtl sim with above instructions (and generate a 'ram_path.sv')
        self.__run_rtl_sim(hex_py2rtl_path, hex_rtl2py_path)

        # 6. load hex generated by RTL sim
        if target == 'SIM':
            sdram_rtlsim, wram_rtlsim, fram0_rtlsim, fram1_rtlsim = self.__load_hex_all(hex_rtl2py_path)
        elif target == 'ASIC':
            sdram_rtlsim = self.__load_mem_hex(hex_rtl2py_path, 'sdram', int(self.hw_cstr['sdram']/4), 4, 512, 16)
        else:
            raise ValueError('Unsupported RTL SIM TARGET')

        # 7. check if data is consistent
        print('########################')
        if (self.mem['sdram']==sdram_rtlsim).all() == False:
            print('Validation failed. SDRAM inconsistency detected.')
            self.__dump_mem_hex(self.mem['sdram'], hex_val_path,'sdram', int(self.hw_cstr['sdram']/4), 4, 512, 16)
            return
        if target == 'SIM':
            if (self.mem['wram']==wram_rtlsim).all() == False:
                print('Validation failed. WRAM inconsistency detected.')
                self.__dump_mem_hex(self.mem['wram'], hex_val_path, 'wram', int(self.hw_cstr['wram']*1024/4), 1, 1, 32)
                return
            if (self.mem['fram0']==fram0_rtlsim).all() == False:
                print('Validation failed. FRAM0 inconsistency detected.')
                self.__dump_mem_hex(self.mem['fram0'], hex_val_path, 'fram0', int(self.hw_cstr['fram']*1024/16), 4, 1, 32)
                return
            if (self.mem['fram1']==fram1_rtlsim).all() == False:
                print('Validation failed. FRAM1 inconsistency detected.')
                self.__dump_mem_hex(self.mem['fram1'], hex_val_path, 'fram1', int(self.hw_cstr['fram']*1024/16), 4, 1, 32)
                return
        
        print('Validation passed!')
        print('########################')

    ## Main MCU Validations
    def validate_mcu(self, hwb_end, hwb_start={'layer_id': 'Conv2d_0', 'hwb_id': 0}):
        """
        generate arrays of weights/img/inst/data checkpoint
        """
        headers_py2mcu_path = os.path.abspath(os.path.join(self.run_dir, 'headers_py2mcu'))
        if os.path.exists(headers_py2mcu_path) == False:
            os.mkdir(headers_py2mcu_path)

        # 0. init sdram (inlcuding image and net params) and prepare 
        self.__init_sdram()
        w_fc, b_fc = self.__prepare_fc_params()

        # 1. dump params, img, inst into const arrays 
        # params: 826032B conv params, start from flash 256KB to 1152KB, SDRAM 0 to buf0_ost
        self.__dump_mem_header(self.mem['sdram'], 'int8', headers_py2mcu_path, 'params_conv', 0, self.param_conv_byte_size, '0x08040000')
        # params: 512x1001 fc weights, 1001 fc biases (in float32). start from flash 1152KB to 1664KB
        self.__dump_mem_header(w_fc, 'int8', headers_py2mcu_path, 'w_fc', 0, w_fc.shape[0], '0x08120000')
        self.__dump_mem_header(b_fc, 'float32', headers_py2mcu_path, 'b_fc', 0, b_fc.shape[0], '0x0819d200')

        # inst: 32KB, start from flash 1664KB to 1696KB
        self.__dump_inst_header(header_dir=headers_py2mcu_path, hwb_start=hwb_start, hwb_end=hwb_end, 
            flash_base_addr='0x081a0000', append_int=True, use_hw_blocks_opt=True)
        
        # img: 160x160x3, can be stored in MCU on-chip sram
        self.__dump_mem_header(self.mem['sdram'], 'int8', headers_py2mcu_path, 'img', self.mem_map['buf0_ost'], self.img_byte_size, None)


        # 3. run behavior level sim from initial to hwb_start
        # self.__run_behavior_sim(hwb_start=hwb_start, hwb_end=hwb_end, dump_tensor=True)

        # print(self.mem['sdram'][self.mem_map['buf1_ost']:self.mem_map['buf1_ost']+4])

        # # 5. run rtl sim with above instructions (and generate a 'ram_path.sv')
        # self.__run_rtl_sim(hex_py2rtl_path, hex_rtl2py_path)

    def validate_non_conv_layers(self): 
        # 1. get the tensors after the last conv layer
        # ifm: [-8,8), w: [-1,1), b: float
        ifm, w, b, ofm = self.__get_tensors([
            'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/act_quant/FakeQuantWithMinMaxVars',
            'MobilenetV1/Logits/Conv2d_1c_1x1/weights_quant/FakeQuantWithMinMaxVars',
            'MobilenetV1/Logits/Conv2d_1c_1x1/biases/read',
            'MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd'
        ])
        ifm = np.squeeze(ifm, axis=0)
        w = np.squeeze(w, axis=(0,1))
        ofm = np.squeeze(ofm, axis=(0,1,2))
        
        # 2. get the INT8 tensors
        ifm = self.__quant_tensor(ifm, 4) # log2(128/8)
        w   = self.__quant_tensor(w, 7)   # log2(128/1)

        # 3. calculate the output before softmax
        #   first do avg pooling (the result should be float)
        ifm = np.mean(ifm, axis=(0,1))
        #   then calculate MVM
        ofm_ut = np.dot(ifm,w)
        #   scale the result back (by dividing 2^11)
        ofm_ut = ofm_ut/2048
        #   add bias
        ofm_ut += b
        print(ofm_ut[0:10]*2048)

        # 4. argmax
        print(np.max(ofm_ut))
        print(np.argmax(ofm_ut))

    def test_img(self):
        result = self.__get_tensors(['MobilenetV1/Predictions/Reshape_1'])
        print(np.argmax(result))

if __name__ == '__main__':
    # usage: python validator.py run_dir img_size

    # v = Validator(run_dir=sys.argv[1], img_size=int(sys.argv[2]))
    v = Validator(run_dir='nets/mnet_v1_160_0.5', img_size=160)

    # v.validate_layer_at_layer_level(v.net.layers[1])
    # v.validate_layer_at_tile_level(v.net.layers[10], {'fram':128,'wram':32})
    # v.validate_at_layer_level()
    # v.validate_at_tile_level()
    # v.validate_at_hw_block_level(use_opt_hw_blocks=True)
    # v.validate_params_in_sdram()
    # v.test()
    # v.count_hw_block_num()

    # v.validate_rtl( 
    #                    hwb_start={'layer_id': 'Conv2d_0', 'hwb_id': 0}, 
    #                    hwb_end={'layer_id':  'Conv2d_1_pointwise', 'hwb_id': 0},
    #                    target='ASIC')
    v.validate_rtl_opt()

    # v.validate_at_hw_block_level(use_opt_hw_blocks=False)
    # v.validate_mcu({'layer_id':  'Conv2d_14_depthwise', 'hwb_id': 0})
    # print(v.mem['sdram'][v.mem_map['buf0_ost']: v.mem_map['buf0_ost']+40])
    # print(v.mem['sdram'][v.mem_map['buf1_ost']: v.mem_map['buf1_ost']+40])
    # v.test_img()