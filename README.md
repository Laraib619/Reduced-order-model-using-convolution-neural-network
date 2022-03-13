# Reduced-order-model-using-convolution-neural-network Auto-Encoder.
## Problem Description
The focus of this work is to develop a model which can extract dominant spatial & temporal modes from 2D laminar flow and use its property and pattern to define the high dimensional flow and then to reconstruct the original flow field. Mean square error to be decreased which is basically a loss occurred during reconstruction of the model. The training and test data which are primarily snapshots obtained from the numerical inquisition of a 2D opposed jet impinging on laminar flow in a rectangular channel by two impinging jets are of different non-reacting miscible fluids at different temperature interacting in a mixed convection regime. 
For the purpose of training and testing, 2000 snapshot captured over sufficient number of cycles for a periodic flow data, so as to capture all the flow features, is used as the high dimensional data. It is used to extract its modes using convolution neural network and mode decomposition convolution neural network technique and reconstruct it to its original form.
### Objectives
(a)	Compare different activation functions of convolution neural network and mode decomposition convolution neural network for 2D periodic laminar flow and to identify the activation function best suited for this model.
(b)	To reduce the loss during reconstruction
(c)	To reconstruct the flow to original form and compare with reference DNS data. 
