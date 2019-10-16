# Mobilenet-v1 0.5 128 cifar-10

``` bash
# from TF2IR/
python TF2IR.py -i models/mobilenet_v1_0.5_128_cifar10/mobilenet_v1_0.5_128_cifar10.pb -o models/mobilenet_v1_0.5_128_cifar10/ --image_height 128 --image_width 128 --input_tensor_name images:0 --end_points MobilenetV1/Dense/act_quant/FakeQuantWithMinMaxVars:0 --test_image models/mobilenet_v1_0.5_128_cifar10/ship.png
```
