# TF2IR
此Repo实现了由TensorFlow frozen graph向自定义中间表达(Intermediate Representation, IR)转换的工具。该工具读入一个quant-aware-training得到的推理图，将关键层信息提取并储存为json格式，并将参数和测试张量提出并储存为npy格式。

## 项目组成
- `TF2IR.py` 主要转换脚本；
- `models` 存放不同模型；
- `utils` 存放从IR中抽取硬件相关信息的脚本。

## IR的格式定义
### 公用定义
- `prev_layer` 存放前序层的列表，输入张量统一用`input`表示；
- `next_layer` 存放后续层的列表，网络结尾统一用`endpoint`表示；
- 权重及测试张量命名规则为`layername_weight/bias/input/output/add/pl.npy`, 格式为：
    - `conv weight`: HWCiCo
    - `dwconv weight`: HWC
    - `bias`: C
    - `feature map`: HWC

### 算子及关键key定义
#### CONV/DWCONV
- `operation`: `conv`代表二维卷积，`dwconv`代表深度可分离卷积
- `activation_type`: 卷积操作内的激活类型，可取值`Relu6`, `None`
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
    "prev_layer": [
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
- 