# Deeplab based on MobilenetV2-0.5 with 513x513 input, trained on PASCAL VOC

``` bash
# from TF2IR/
python TF2IR.py -i models/deeplab_mobilenetv2_0.5_513x513_VOC/deeplab_mobilenetv2_0.5_513x513_VOC.pb -o models/deeplab_mobilenetv2_0.5_513x513_VOC/ \
--input_tensor_name sub_2:0 --image_height 513 --image_width 513 \
--end_points extra_op_quant_2/FakeQuantWithMinMaxVars:0 \
--test_image models/deeplab_mobilenetv2_0.5_513x513_VOC/motor.jpg \
--preprocess True --input_name_before_preprocess ImageTensor:0
```
