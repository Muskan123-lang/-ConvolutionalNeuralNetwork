# -ConvolutionalNeuralNetwork
-> CNNs are a powerful tool for ML, especially in tasks related to computer vision.
-> CNNs are a specialized class of neural networks designed to effectively process grid-like data, such as images.

Convolutional Neural Network (CNN)
-> It is a type of deep learning algorithm that is particularly well-suited for image recognition and processing tasks.
-> It is made up of multiple layers, including convolutional layers, pooling layers, and fully connected layers. 
-> The architecture of CNNs is inspired by the visual processing in the human brain, and they are well-suited for capturing hierarchical patterns and spatial dependencies within images.
-> Key Components of CNN: 
Convolutional layers, Pooling layers, Activation Functions, Fully Connected layers

* Convolutional Layers: These layers apply convolutional operations to input images, using filters (also known as kernels) to detect features such as edges, textures, and more complex patterns. Convolutional operations help preserve the spatial relationships between pixels.
(Note: A filter, or kernel, in a CNN is a small matrix of weights that slides over the input data (such as an image), performs element-wise multiplication with the part of the input it is currently on, and then sums up all the results into a single output pixel. This process is known as convolution.)

** Pooling Layers: Pooling layers downsample the spatial dimensions of the input, reducing the computational complexity and the number of parameters in the network. Max pooling is a common pooling operation, selecting the maximum value from a group of neighboring pixels.

*** Activation Functions: Non-linear activation functions, such as Rectified Linear Unit (ReLU), introduce non-linearity to the model, allowing it to learn more complex relationships in the data.
(Note: Activation Functions An artificial neuron calculates the ‘weighted sum’ of its inputs and adds a bias. Mathematically, 
![Screenshot 2024-04-25 193635](https://github.com/Muskan123-lang/-ConvolutionalNeuralNetwork/assets/68841119/079fb73d-7063-4a19-8912-4deacfa67f1a)
)
(Note: ReLU: The ReLU function is the Rectified linear unit. It is the most widely used activation function. It is defined as: f(x) = max(0, x) 
The main advantage of using the ReLU function over other activation functions is that it does not activate all the neurons at the same time. It means, if you look at the ReLU function, if the input is negative it will convert it to zero and the neuron does not get activated.)

![Screenshot 2024-04-25 192236](https://github.com/Muskan123-lang/-ConvolutionalNeuralNetwork/assets/68841119/7bd29ec9-a2a2-402c-acd4-be0116178818)

**** Fully Connected Layers: These layers are responsible for making predictions based on the high-level features learned by the previous layers. They connect every neuron in one layer to every neuron in the next layer.


-> CNNs are trained using a large dataset of labeled images, where the network learns to recognize patterns and features that are associated with specific objects or classes. 
-> Proven to be highly effective in image-related tasks, achieving state-of-the-art performance in various computer vision applications.
-> Their ability to automatically learn hierarchical representations of features makes them well-suited for tasks where the spatial relationships and patterns in the data are crucial for accurate predictions. 
-> CNNs are widely used in areas such as image classification, object detection, facial recognition, and medical image analysis.

(The convolutional layers are the key component of a CNN, where filters are applied to the input image to extract features such as edges, textures, and shapes.
The output of the convolutional layers is then passed through pooling layers, which are used to down-sample the feature maps, reducing the spatial dimensions while retaining the most important information. The output of the pooling layers is then passed through one or more fully connected layers, which are used to make a prediction or classify the image.)



