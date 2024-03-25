# **Build the Neural Network using MNIST dataset using Pytorch Framewrok**
## Why Pytorch
PyTorch is favored for research and dynamic projects,Pythonic approach, ideal for beginners and rapid prototyping so pytorch is used
## Why MNIST dataset
The MNIST handwritten digit classification problem is a standard dataset used in computer vision and deep learning.

Although the dataset is effectively solved, it can be used as the basis for learning and practicing how to develop, evaluate, and use convolutional deep learning neural networks for image classification from scratch.

# **Import Libraries** (Block 1)
Here we are importing all the packages required for building the neural network
**Torch package**
The torch package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors.
**Torch nn**
This contains different classess that help to build neural network models.
**Torch funtional**
The functional API of PyTorch is a powerful tool that enables you to write high-performance neural network mo
dels
**Torch optim**
torch.optim is a package implementing various optimization algorithms.
**Torchvision**
Torchvision provides additional functionalities to manipulate and process images with standard image processing algorithms. It has the computer vision models and datasets
1.   datasets:
        It has common datasets like MNIST, CIFAR10, ImageNet etc.
2.  transforms
       Torchvision supports common computer vision transformations in the torchvision.transforms and torchvision.transforms.v2 modules. Transforms can be used to transform or augment data for training or inference of different tasks (image classification, detection, segmentation, video classification).
# **Data Transformations (Block 2)**
Data transformation is also known as data preparation or data preprocessing. There are lots of different names for the same thing. It makes sure that your data is clean and ready to be used by your machine learning algorithm. Without data transformation, your AI won't be able to make accurate predictions.
**To Tensor**
This is a very commonly used conversion transform. In PyTorch, we mostly work with data in the form of tensors. If the input data is in the form of a NumPy array or PIL image, we can convert it into a tensor format using ToTensor.

# Dataset and Creating Train/Test Split (Block 3)
The datasets used here is the MNIST dataset.
Inside of it is a Raw folder and a Processed folder.
Inside of the Raw folder, we see the four files that were downloaded.
Inside of the Processed folder, we see the two files that were generated after the processing.
The train parameter is set to false because we want test set, not the train set.
Then like the training set, we set download to true and transform to none.
The train dataset will be 60,000 and test dataset is 10,000
Note that because we set the transform parameter to none, that they should be what comes out of the raw data.
We use the root parameter to define where to save the data.
The train parameter is set to true because we are initializing the MNIST training dataset.
The download parameter is set to true because we want to download it if it’s not already present in our data folder.
The transform parameter is set to we  want to apply any image manipulation transforms .
# **Dataloader Arguments & Test/Train Dataloaders(Block 4)**
# What Does a PyTorch DataLoader Do?
The PyTorch DataLoader class is a tool which help to prepare, manage, and serve the data to the deep learning networks. Because many of the pre-processing steps we will need to do before beginning training a model, finding ways to standardize these processes is critical for the readability and maintainability of your code.

The PyTorch DataLoader used below are :
**Define a dataset to work with:** identifying where the data is coming from and how it should be accessed.

**Batch the data:**
 To define how many training or testing samples to use in a single iteration. Because data are often split across training and testing sets of large sizes, being able to work with batches of data can allow us  training and testing processes to be more manageable.
**Shuffle the data:** 
PyTorch can handle shuffling data for us as it loads data into batches. This can increase representativeness in the dataset and prevent accidental skewness.
**Transforms:**
The transform() method allows you to execute a function for each value of the DataFrame.
Tensor image are expected to be of shape (C, H, W), where C is the number of channels, and H and W refer to height and width. Most transforms support batched tensor input. A batch of Tensor images is a tensor of shape (N, C, H, W), where N is a number of images in the batch. The v2 transforms generally accept an arbitrary number of leading dimensions (..., C, H, W) and can handle batched images or batched videos.
**Normalization**
The goal of normalization is to transform features to be on a similar scale. This improves the performance and training stability of the model
Reference for Normalization: [Normalization for different datsets](https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457)
**Torch Summary**
This can be used to print out the trainable and non-trainable parameters in a Keras-like manner for PyTorch models.
*Torch cuda**
It is used to set up and run CUDA operations. It keeps track of the currently selected GPU, and all CUDA tensors you allocate will by default be created on that device.
**Torch utils**
TorchUtils is a Python package providing helpful utility APIs for the PyTorch projects.
# **The model**
We have 7 convolution layers and one linear layer. We are using dropout and average pooling in the model for better accuracy
