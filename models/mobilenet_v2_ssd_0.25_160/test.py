import numpy as np 
from numpy.lib.stride_tricks import as_strided

def conv2d(ifm, w, b, padding, stride, b_ls, ofm_rs):
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
    ifm = insert_padding(ifm, padding)
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
    # 0. add half of OFM LSB to the ofm before right shift
    ofm += (1<<(ofm_rs-1))
    ofm = np.right_shift(ofm, ofm_rs)
    return relu6(ofm)

def insert_padding(fm, padding):
    """
    padding: [T, L, B, R]
    """
    fm = np.pad(fm, ((padding[0], padding[2]),(padding[1], padding[3]),(0,0)), 'constant')
    return fm

def relu6(fm):
    fm[fm<0] = 0
    fm[fm>96] = 96
    return fm

def add_rounding_to_bias(b_quantized, b_ls, ofm_rs):
    """
    add half of OFM LSB to bias, so that the hardware can only perform shifting
    and truncate the bias if the b_ls_raw is negative

    returns:
    b, b_shift
    """
    shift = ofm_rs-1-b_ls
    if shift >= 0:
        b_quantized += (1<<shift)

    return b_quantized


layer_name = 'FeatureExtractor_MobilenetV2_Conv_Conv2D_Fold'
ifm = np.load('params/'+layer_name+'_input.npy')
w = np.load('params/'+layer_name+'_weight.npy')
b = np.load('params/'+layer_name+'_bias.npy')
ofm = np.load('params/'+layer_name+'_output.npy')

padding = [0,0,1,1]
stride = 2
b_ls = 7
ofm_rs = 7

# add half to bias for replacing rounding with shifting
# b = add_rounding_to_bias(b, b_ls, ofm_rs)

ofm_manual = conv2d(ifm, w, b, padding, stride, b_ls, ofm_rs)

diff = ofm_manual-ofm
print((diff==0).all())
assert (diff==0).all()
