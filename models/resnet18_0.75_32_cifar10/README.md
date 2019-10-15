# resnet-18 0.75 32 cifar10

``` bash
# from TF2IR/
python TF2IR.py -i models/resnet18_0.75_32_cifar10/resnet18_0.75_32_cifar10.pb -o models/resnet18_0.75_32_cifar10/ --image_height 32 --image_width 32 --input_tensor_name images:0 --end_points resnet_v1_18/logits/act_quant/FakeQuantWithMinMaxVars:0 --test_image models/resnet18_0.75_32_cifar10/ship.png
```
