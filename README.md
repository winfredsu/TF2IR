# TF2IR
Find valid ops in the graph and convert them to layer representations. 


## Conversion Rules
1. IR layer name = OP name in the frozen graph


## How to determine if an OP is a valid layer that should be extracted?
### Reshape
A valid reshape should have two inputs: one from valid ops and the other as a const.

### Conv2D/DWConv
All such ops should be a valid layer. 

### Add


## Additional Information for Test