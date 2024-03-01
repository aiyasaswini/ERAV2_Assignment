## The points to consider while constructing a  model
# How many layers
No of layers to be used depends on input dataset and also on the size of the images.

# Max Pooling
It is a mechanism which reduces the receptive field .
It is the maximum no taken as output for the given values
# 1x1 Convolutions
It is used to reduce the number of channels.It is a single dimension matrix
# 3x3 Convolutions
It is used as a filter .we can create 5X 5 or 7X 7 filter by using this as base
# Receptive Field
How much the input is seen in the layers
# SoftMax
It is a better represenatation of the output values
# Learning Rate
It is the hyperparameter we adjust

# Kernels and how do we decide the number of kernels?
It depends on the input image size
# Batch Normalization
To normalize the data so that all fit in same scale
# Image Normalization
To divide RGB values by 255 to normalize the image

# Position of MaxPooling
The position of Maxpooling should be 2 or more layers above the final layer
# Concept of Transition Layers
The layers in between two layers
# Position of Transition Layer
It will be in between two layers
# DropOut
To knowingly droping some neurons
# When do we introduce DropOut, or when do we know we have some overfitting
This is used to reduce overfitting a network.Overfitting can be noticed when training accuracy and validation accuracy difference is high

# The distance of MaxPooling from Prediction
at least 3 layers
# The distance of Batch Normalization from Prediction
Before the final layer
# When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
When the memory is not a contrained
# How do we know our network is not going well, comparatively, very early
If the validation accuracy is giving same value for many layers
# Batch Size, and Effects of batch size
If we increase batch size the accuracy may reduce