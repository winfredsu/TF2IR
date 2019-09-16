import numpy as np

def quant_tensor(tensor, scale):
    return (np.power(2,scale)*tensor).astype(np.int32)

def add_rounding_to_bias_and_truncate(b, b_ls_raw, ofm_rs):
    """
    add half of OFM LSB to bias, so that the hardware can only perform shifting without rounding
    and truncate the bias if the b_ls_raw is negative
    """
    shift = ofm_rs-1-b_ls_raw
    if shift >= 0:
        b += (1<<shift)
    if b_ls_raw < 0:
        b = np.right_shift(b,-b_ls_raw)

    return b

def tensor2mem(mode, tensor):
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
        return(tensor2mem('w_2d_s1', tensor_reordered))
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
        # mem shape: Ci-{147x, 258x, 369x}
        if tensor.shape[0]!=3 or tensor.shape[1]!=3:
            raise ValueError('kernel size must be 3x3 for w_dw_s2')
        
        tensor_reordered = np.zeros(tensor.shape).astype(np.int)
        tensor_reordered[:,0,:,:] = tensor[:,0,:,:]
        tensor_reordered[:,1,:,:] = tensor[:,2,:,:]
        tensor_reordered[:,2,:,:] = tensor[:,1,:,:]
        return(tensor2mem('w_dw_s1', tensor_reordered))       
    elif mode == 'w_dw_s1':
        # tensor shape: HWCiCo=1
        # mem shape: Ci-{147x, 258x, 369x}
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