IR定义主要修改：
## 总体
1. 修改json文件为一个有序列表，每个元素即为层定义。该有序列表可保证每层只依赖其前序元素，但暂不能保证内存利用最优。
2. 合法的层包括conv/dwconv/concat/reshape
3. 在params中添加每层的input/output tensor用于**测试**，文件名为`layer_name_ifm/ofm.raw`(或layer_name_pl/add/ofm.raw, 对于add层), 并附测试图片。测试时注意:
    - C和TensorFlow中rounding的差异
    - 输入像素应为[-128,127]的整数
4. next_layer为['endpoint']代表这是一个输出节点，可能不止一个
5. raw文件顺序统一定义为：
    - weight: HWCiCo (conv), HWC (dwconv)
    - feature map: HWC

## Conv/DWconv
1. Depthwise Conv层operation值为`dwconv`
2. 由于硬件需要conv和relu打包成一个算子，Conv/DWconv层新增名为`activation_type`的键, 可选值为`None`和`Relu6`.

## Add
1. 新增Feature Map Size信息， 即input_size和input_channel_num
2. 由于Add操作后也可能出现relu, 新增名为`activation_type`的键，定义同conv中的描述。
3. 顺序定义: pl对应previous_layer[0]， add对应previous_layer[1]
4. pl_shift_bit和add_shift_bit实际代表该tensor相对128的scale factor