# TF2IR
此Repo实现了由TensorFlow frozen graph向自定义中间表达(Intermediate Representation, IR)转换的工具。该工具读入一个quant-aware-training得到的推理图，将关键层信息提取并储存为json格式，并将参数和测试张量提出并储存为npy格式。

## 1. 项目组成
- `TF2IR.py` 主要转换脚本；
- `models` 存放不同模型；
- `utils` 存放从IR中抽取硬件相关信息的脚本。

## 2. 使用方法
`python TF2IR -i path_to_graph.pb -o output_dir --image_height 160 --image_width 160 --input_tensor_name normalized_input_image_tensor:0 --end_points reshape:0 reshape:1 reshape:2 --test_image test.jpg`
- `-i`: 输入frozen graph路径
- `-o`: 输出路径(应为directory)
- `--image_height`: 图片高度
- `--image_width`: 图片宽度
- `--input_tensor_name`: 推理图中的输入Tensor名称，例如：`input:0`
- `--end_points`: 推理图的结束Tensor名称，例如：对于SSD类检测网络，将各检测头的`Reshape:0`作为结束点，提供给SSD使用。
- `--test_image`: 测试图片路径，用于生成每层的输入输出张量

## 3. 支持的图结构及主要限制
- 目前所有input/w/b/output均为int8格式，采用power-of-2范围的对称量化，即将原tensor取power-of-2范围后，均匀放缩到[-128, 127]范围内。
- 推理图中不能出现Batchnorm单元，所有Batchnorm在训练时都应该Fold进权重。
- CONV/DWCONV在TensorFlow Graph中的结构应为如下格式。其中weight和bias均来源于Fake Quant单元；输出经过Relu6激活或无激活函数；最终输出由Fake Quant单元得到。
![quantized_conv_block](./doc/quantized_conv_block.png)
- ADD在TensorFlow Graph中的结构应为如下格式。其中x和y来自前序层的输出Fake Quant单元；输出经过Relu6激活或无激活函数；最终输出由Fake Quant单元得到。
![quantized_add_block](./doc/quantized_add_block.png)
- 该工具能够自动识别并跳过图中的Identity单元。
- 在TensorFlow的推理过程中，输入是[-1,1]范围的浮点数；在输出的IR中，第一层的输入log2(scale)信息已被用于移位信息的计算，因此网络的输入应为[-128,127]的整数。


## 4. IR的格式定义
### 4.1. 公用定义
- `previous_layer` 存放前序层的列表，输入张量统一用`input`表示；
- `next_layer` 存放后续层的列表，网络结尾统一用`endpoint`表示；
- 权重及测试张量命名规则为`layername_weight/bias/input/output/add/pl.npy`, 存储格式为：
    - `conv weight`: HWCiCo
    - `dwconv weight`: HWC
    - `bias`: C
    - `feature map`: HWC

### 4.2. 支持的算子及其关键key定义
#### CONV/DWCONV
- `operation`: `conv`代表二维卷积，`dwconv`代表深度可分离卷积
- `activation_type`: 卷积操作内的激活类型，可取值`Relu6`, `None`
- `dilations`: 对于普通卷积，该项应为[1,1], 对于空洞卷积(dilated/atrous conv), 该项为H/W方向的dilation. 注意对于空洞卷积操作，strides必须为1. 此外，Tensorflow.slim库对空洞卷积的处理方式为(space2batch-conv_with_dilation1-batch2space), 此IR中的padding信息根据原有space2batch和batch2space计算得到。
- `xx_log2scale`: 该tensor对应的log2(scale), 例如：某input_tensor的实际范围为[-4,4), 使用8bit量化，则对应的`input_log2scale=5`
- `output_shift`: 中间结果->输出的右移位数，`output_shift=input_log2scale+weight_log2scale-output_log2scale`
- `bias_shift`: 加偏置时的偏置左移位数，`bias_shift=input_log2scale+weight_log2scale-bias_log2scale`
- `load_bias`: 取`true`或`false`, 代表该层是否需要加偏置
- `xx_dtype`: 数据类型，目前都应为`int8`

``` json
  {
    "name": "FeatureExtractor_MobilenetV2_Conv_Conv2D_Fold",
    "operation": "conv",
    "activation_type": "Relu6",
    "input_log2scale": 7,
    "weight_log2scale": 4,
    "bias_log2scale": 4,
    "output_log2scale": 4,
    "output_shift": 7,
    "bias_shift": 7,
    "load_bias": true,
    "input_channel_num": 3,
    "output_channel_num": 16,
    "input_size": {
      "height": 160,
      "width": 160
    },
    "padding": {
      "top": 0,
      "bottom": 1,
      "left": 0,
      "right": 1
    },
    "stride": {
      "height": 2,
      "width": 2
    },
    "dilations": {
      "height": 1, 
      "width": 1
    },
    "kernel_size": {
      "height": 3,
      "width": 3
    },
    "output_size": {
      "height": 80,
      "width": 80
    },
    "input_dtype": "int8",
    "output_dtype": "int8",
    "weight_dtype": "int8",
    "bias_dtype": "int8",
    "previous_layer": [
      "input"
    ],
    "next_layer": [
      "FeatureExtractor_MobilenetV2_expanded_conv_depthwise_depthwise_Fold",
      "FeatureExtractor_MobilenetV2_expanded_conv_add"
    ]
  }
```

#### ADD
- `activation_type`: 卷积操作内的激活类型，可取值`Relu6`, `None`
- `pl_xx`: 代表被加数，即TensorFlow图中的op.inputs[0]
- `add_xx`: 代表加数， 即TensorFlow图中的op.inputs[1]
- `output_shift_bit`: 中间累加结果->输出的**左移**位数, `output_shift_bit = output_log2scale - min(add_log2scale, pl_log2scale)`, 负数代表右移。
- `add_shiftbit`和`pl_shiftbit`: 历史遗留命名，含义与`add_log2scale`和`pl_log2scale`相同。
- `dtype`: 数据类型，目前都应为`int8`

``` json
  {
    "name": "FeatureExtractor_MobilenetV2_expanded_conv_add",
    "operation": "add",
    "activation_type": "None",
    "input_channel_num": 16,
    "input_size": {
      "height": 80,
      "width": 80
    },
    "dtype": "int8",
    "pl_log2scale": 4,
    "pl_shiftbit": 4,
    "add_log2scale": 4,
    "add_shiftbit": 4,
    "output_log2scale": 3,
    "output_shift_bit": -1,
    "pl_name": "FeatureExtractor_MobilenetV2_expanded_conv_project_Conv2D_Fold",
    "add_name": "FeatureExtractor_MobilenetV2_Conv_Conv2D_Fold",
    "previous_layer": [
      "FeatureExtractor_MobilenetV2_Conv_Conv2D_Fold",
      "FeatureExtractor_MobilenetV2_expanded_conv_project_Conv2D_Fold"
    ],
    "next_layer": [
      "FeatureExtractor_MobilenetV2_expanded_conv_1_expand_Conv2D_Fold"
    ]
  }
```

#### RESHAPE
- `dtype`: 数据类型，目前都应为`int8`
- `input/output_shape`: 输入/输出shape(不定长)
- `reshape_param`: reshape参数
- `log2scale`: 输入/输出tensor的log2(scale)

``` json
  {
    "name": "BoxPredictor_0_Reshape_1",
    "operation": "reshape",
    "reshape_param": [
      -1,
      2
    ],
    "input_shape": [
      10,
      10,
      6
    ],
    "output_shape": [
      300,
      2
    ],
    "dtype": "int8",
    "log2scale": 2,
    "previous_layer": [
      "BoxPredictor_0_ClassPredictor_Conv2D_Fold"
    ],
    "next_layer": [
      "endpoint"
    ]
  }
```

#### AVGPOOL/MAXPOOL
- `dtype`: 数据类型，目前都应为`int8`
- `kernel_size`: average pooling范围参数
- `input_log2scale`, `output_log2scale`: 输入和输出的scale, 由于average pooling操作会改变tensor范围，因此二者可能不同（`input_log2scale`小于`output_log2scale`)
- `input_pre_ls`: 由于存在上述问题，可将输入的整数提前左移后求平均再取整，该键定义了“提前左移”的位移量
- `input_size`, `stride`, `padding`, `kernel_size`与`output_size`的关系：
  
``` json
  {
    "name": "resnet_v1_18/AvgPool",
    "operation": "avg_pool",
    "input_channel_num": 64,
    "input_size": {
      "height": 6,
      "width": 6
    },
    "input_log2scale": 4,
    "input_dtype": "int8",
    "kernel_size": {
      "height": 6,
      "width": 6
    },
    "stride": {
      "height": 2,
      "width": 2
    },
    "padding": {
      "top": 0,
      "bottom": 1,
      "left": 0,
      "right": 1
    },
    "output_channel_num": 64,
    "output_size": {
      "height": 1,
      "width": 1
    },
    "output_log2scale": 5,
    "output_dtype": "int8",
    "input_pre_ls": 1, 
    "previous_layer": [
      "some_layers"
    ],
    "next_layer": [
      "FeatureExtractor_MobilenetV2_expanded_conv_add"
    ]
  }
```

#### ResizeBilinear
双线性插值，用于改变feature map大小
- `dtype`: 数据类型，目前都应为`int8`
- `align_corners`: 参考tensorflow resizebilinear的说明
- `input_log2scale`, `output_log2scale`: 输入和输出的scale, 由于某些未知原因二者可能不同

#### ConcatV2
目前仅支持二输入的concat, 在H/W/C方向上进行concat.
- `dim`: concat方向
- `input0_log2scale`, `input1_log2scale`, `output_log2scale`: 两个输入和输出的scale, 可能不同，可通过预移位处理。

```json
  {
    "name": "concat",
    "operation": "concat2",
    "dim": "C",
    "input0_channel_num": 256,
    "input0_size": {
      "height": 65,
      "width": 65
    },
    "input0_log2scale": 5,
    "input0_dtype": "int8",
    "input1_channel_num": 256,
    "input1_size": {
      "height": 65,
      "width": 65
    },
    "input1_log2scale": 4,
    "input1_dtype": "int8",
    "output_channel_num": 512,
    "output_size": {
      "height": 65,
      "width": 65
    },
    "output_log2scale": 4,
    "output_dtype": "int8",
    "previous_layer": [
      "aspp0_Conv2D_Fold",
      "ResizeBilinear"
    ],
    "next_layer": [
      "concat_projection_Conv2D_Fold"
    ]
  },


```



## 5. 解析原理
### 5.1 找到input tensor
根据input_tensor_name找到input tensor

### 5.2 维护tensors_ready, ops_discovered和info_prev_layers三个集合
- `tensors_ready`: 已处理完的tensor
- `ops_discovered`: 每处理完一个tensor之后，寻找其real consumer，并加入ops_discovered. 
- `info_prev_layers`: 每处理完一个layer后，将其consumer的previous_layer标注为该layer

### 5.3 对图进行有依赖的DFS
从ops_discovered队列末尾向前，直到找到一个op, 其dependency均已出现在tensors_ready之中。从ops_discovered中取出这个op.

### 5.4 根据不同的op类型，提取layer信息
以卷积算子为例，首先找到卷积层真正的ifm/ofm/w/b (fake quant op), 再从卷积算子中提取卷积参数。此后对ifm/ofm/w/b tensor进行求值，并得到中间feature map的size信息。然后从各fake quant op中提取量化宽度和scale信息，并计算bias和输出的移位信息，同时根据量化和scale信息换算出量化后的tensor值。 最终，将layer信息存入字典，并保存量化过的tensor文件。

### 5.5 处理previous/next信息
首先将该层的输出tensor加入到tensors_ready中；然后从info_prev_layers中找到该层的前序层信息，并加入layer字典中；此后，若该层的输出tensor非结束点，找到其所有真正的consumer, 并加入next_layer信息，更新ops_discovered和info_prev_layer. 