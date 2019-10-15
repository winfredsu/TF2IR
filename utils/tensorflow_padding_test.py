import tensorflow as tf
import numpy as np

# def input_4_stride_2_kernel_1():
"""
input: (4x1) [1.0, 2.0, 3.0, 4.0]
kernel: (1x1) [1.0]
stride: 2
output: (2x1) 

Use this test to see tensorflow padding rule when stride is larger than kernel size
"""
i = tf.constant((np.ones(4) + 2*np.arange(4)).reshape(1,4,1,1), dtype=tf.float32, name='input')
f = tf.constant(np.ones(3).reshape(3,1,1,1), dtype=tf.float32, name='filter')

conv = tf.nn.conv2d(input=i, filter=f, strides=(1,2,1,1), padding='SAME')
with tf.Session() as sess:
    out = sess.run(conv)
    print(out)