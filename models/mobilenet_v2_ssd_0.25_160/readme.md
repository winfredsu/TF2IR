# IR定义主要修改
## 总体
1. 修改json文件为一个有序列表，每个元素即为层定义。该有序列表可保证每层只依赖其前序元素，但暂不能保证内存利用最优。
2. 合法的层包括conv/dwconv/reshape
3. 在params中添加每层的input/output tensor用于**测试**，文件名为`layer_name_input/output.raw`(或layer_name_pl/add/output.raw, 对于add层), 并附测试图片。测试时注意:
    - C和TensorFlow中rounding的差异
    - 输入像素应为[-128,127]的整数
4. next_layer为['endpoint']代表这是一个输出节点，输出节点可能不止一个
5. npy文件顺序统一定义为：
    - weight: HWCiCo (conv), HWC (dwconv)
    - feature map: HWC
6. Concat操作存在输入scale不一致的问题，暂时不实现，直接将concat前的点作为endpoint. Concat并入CPU部分的检测头实现。

## Conv/DWconv
1. Depthwise Conv层operation值为`dwconv`
2. 由于硬件需要conv和relu打包成一个算子，Conv/DWconv层新增名为`activation_type`的键, 可选值为`None`和`Relu6`.
3. 添加input/output/w/b的log2scale，目前没用到

## Add
1. 新增Feature Map Size信息， 即input_size和input_channel_num
2. 由于Add操作后也可能出现relu, 新增名为`activation_type`的键，定义同conv中的描述。
4. pl_shift_bit和add_shift_bit**重命名**为pl_log2scale和add_log2scale, 添加output_log2scale(目前没用)
5. output_shift_bit = output_log2scale - min(pl_log2scale, add_log2scale)

## Reshape
``` 
  {
    "name": "BoxPredictor_5_Reshape_1",
    "operation": "reshape",
    "reshape_param": [
      -1,
      2
    ],
    "input_shape": [
      1,
      1,
      12
    ],
    "output_shape": [
      6,
      2
    ],
    "dtype": "int8",
    "log2scale": 2,
    "prev_layer": [
      "BoxPredictor_5_ClassPredictor_Conv2D_Fold"
    ],
    "next_layer": [
      "endpoint"
    ]
  }
```
1. `input_shape`，`output_shape`定义为数组，因为其长度不定。