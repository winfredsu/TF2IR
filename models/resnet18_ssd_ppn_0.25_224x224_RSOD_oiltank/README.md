# resnet-18 ssd-ppn 0.25 224x224 trained on RSOD oiltanks

``` bash
# from TF2IR/
python TF2IR.py -i models/resnet18_ssd_ppn_0.25_224x224_RSOD_oiltank/graph.pb -o models/resnet18_ssd_ppn_0.25_224x224_RSOD_oiltank/ --image_height 224 --image_width 224 --input_tensor_name normalized_input_image_tensor:0 \
--end_points WeightSharedConvolutionalBoxPredictor/Reshape:0 \
WeightSharedConvolutionalBoxPredictor_1/Reshape:0 \
WeightSharedConvolutionalBoxPredictor_2/Reshape:0 \
WeightSharedConvolutionalBoxPredictor_3/Reshape:0 \
WeightSharedConvolutionalBoxPredictor_4/Reshape:0 \
WeightSharedConvolutionalBoxPredictor_5/Reshape:0 \
WeightSharedConvolutionalBoxPredictor/Reshape_1:0 \
WeightSharedConvolutionalBoxPredictor_1/Reshape_1:0 \
WeightSharedConvolutionalBoxPredictor_2/Reshape_1:0 \
WeightSharedConvolutionalBoxPredictor_3/Reshape_1:0 \
WeightSharedConvolutionalBoxPredictor_4/Reshape_1:0 \
WeightSharedConvolutionalBoxPredictor_5/Reshape_1:0 \
--test_image models/resnet18_ssd_ppn_0.25_224x224_RSOD_oiltank/test.jpg
```
