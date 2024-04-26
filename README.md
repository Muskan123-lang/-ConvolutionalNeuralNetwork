# ConvolutionalNeuralNetwork
-> CNNs are a powerful tool for ML, especially in tasks related to computer vision.
-> CNNs are a specialized class of neural networks designed to effectively process grid-like data, such as images.

# Convolutional Neural Network (CNN)
-> It is a type of deep learning algorithm that is particularly well-suited for image recognition and processing tasks.
-> It is made up of multiple layers, including convolutional layers, pooling layers, and fully connected layers. 
-> The architecture of CNNs is inspired by the visual processing in the human brain, and they are well-suited for capturing hierarchical patterns and spatial dependencies within images.

# Key Components of CNN: 
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



# Convolutional Neural Network Design
The construction of a convolutional neural network is a multi-layered feed-forward neural network, made by assembling many unseen layers on top of each other in a particular order.
It is the sequential design that give permission to CNN to learn hierarchical attributes.
In CNN, some of them followed by grouping layers and hidden layers are typically convolutional layers followed by activation layers.
The pre-processing needed in a ConvNet is kindred to that of the related pattern of neurons in the human brain and was motivated by the organization of the Visual Cortex.
Convolutional Neural Network Training
CNNs are trained using a supervised learning approach. This means that the CNN is given a set of labeled training images. The CNN then learns to map the input images to their correct labels.

# The training process for a CNN involves the following steps:

Data Preparation: The training images are preprocessed to ensure that they are all in the same format and size.

Loss Function: A loss function is used to measure how well the CNN is performing on the training data. The loss function is typically calculated by taking the difference between the predicted labels and the actual labels of the training images.

Optimizer: An optimizer is used to update the weights of the CNN in order to minimize the loss function.

Backpropagation: Backpropagation is a technique used to calculate the gradients of the loss function with respect to the weights of the CNN. The gradients are then used to update the weights of the CNN using the optimizer.



# CNN Evaluation
After training, CNN can be evaluated on a held-out test set. A collection of pictures that the CNN has not seen during training makes up the test set. How well the CNN performs on the test set is a good predictor of how well it will function on actual data.

The efficiency of a CNN on picture categorization tasks can be evaluated using a variety of criteria. Among the most popular metrics are:

* Accuracy: Accuracy is the percentage of test images that the CNN correctly classifies.
* Precision: Precision is the percentage of test images that the CNN predicts as a particular class and that are actually of that class.
* Recall: Recall is the percentage of test images that are of a particular class and that the CNN predicts as that class.
* F1 Score: The F1 Score is a harmonic mean of precision and recall. It is a good metric for evaluating the performance of a CNN on classes that are imbalanced.

# Different Types of CNN Models
LeNet
AlexNet
ResNet
GoogleNet
MobileNet
VGG

# Applications of CNN
* Image classification: CNNs are the state-of-the-art models for image classification. They can be used to classify images into different categories, such as cats and dogs, cars and trucks, and flowers and animals.

* Object detection: CNNs can be used to detect objects in images, such as people, cars, and buildings. They can also be used to localize objects in images, which means that they can identify the location of an object in an image.

* Image segmentation: CNNs can be used to segment images, which means that they can identify and label different objects in an image. This is useful for applications such as medical imaging and robotics.

* Video analysis: CNNs can be used to analyze videos, such as tracking objects in a video or detecting events in a video. This is useful for applications such as video surveillance and traffic monitoring.

# Advantages of CNN
-> CNNs can achieve state-of-the-art accuracy on a variety of image recognition tasks, such as image classification, object detection, and image segmentation.
-> CNNs can be very efficient, especially when implemented on specialized hardware such as GPUs.
-> CNNs are relatively robust to noise and variations in the input data.
-> CNNs can be adapted to a variety of different tasks by simply changing the architecture of the network.

# Disadvantages of CNN
-> CNNs can be complex and difficult to train, especially for large datasets.
-> CNNs can require a lot of computational resources to train and deploy.
-> CNNs require a large amount of labeled data to train.
-> CNNs can be difficult to interpret, making it difficult to understand why they make the predictions they do.


# Case Study of CNN for Diabetic retinopathy
-> Diabetic retinopathy also known as diabetic eye disease, is a medical state in which destruction occurs to the retina due to diabetes mellitus, It is a major cause of blindness in advance countries.
-> Diabetic retinopathy influence up to 80 percent of those who have had diabetes for 20 years or more.
-> The overlong a person has diabetes, the higher his or her chances of growing diabetic retinopathy.
-> It is also the main cause of blindness in people of age group 20-64.
-> Diabetic retinopathy is the outcome of destruction to the small blood vessels and neurons of the retina.

# Conclusion
Convolutional neural networks (CNNs) are a powerful type of artificial neural network that are particularly well-suited for image recognition and processing tasks. They are inspired by the structure of the human visual cortex and have a hierarchical architecture that allows them to learn and extract features from images at different scales. CNNs have been shown to be very effective in a wide range of applications, including image classification, object detection, image segmentation, and image generation.



# Frequently Asked Questions(FAQs)
1. What is a convolutional neural network (CNN)?
=> A Convolutional Neural Network (CNN) is a type of artificial neural network (ANN) that is specifically designed to handle image data. CNNs are inspired by the structure of the human visual cortex and have a hierarchical architecture that allows them to extract features from images at different scale

2. How does CNN work?
=> CNNs use a series of convolutional layers to extract features from images. Each convolutional layer applies a filter to the input image, and the output of the filter is a feature map. The feature maps are then passed through a series of pooling layers, which reduce their size and dimensionality. Finally, the output of the pooling layers is fed into a fully connected layer, which produces the final output of the network.

3. What are the different layers of CNN?
=> A CNN typically consists of three main types of layers:-
Convolutional layer: The convolutional layer applies filters to the input image to extract local features.
Pooling layer: The pooling layer reduces the spatial size of the feature maps generated by the convolutional layer.
Fully connected layer: The fully connected layer introduces a more traditional neural network architecture, where each neuron is connected to every neuron in the previous layer.

4. What are some of the tools and frameworks for developing CNNs?
=> There are many popular tools and frameworks for developing CNNs, including:
TensorFlow: An open-source software library for deep learning developed by Google.
PyTorch: An open-source deep learning framework developed by Facebook.
MXNet: An open-source deep learning framework developed by Apache MXNet.
Keras: A high-level deep learning API for Python that can be used with TensorFlow, PyTorch, or MXNet.

6. What are some of the challenges of using CNNs?
=> CNNs can be challenging to train and require large amounts of data. Additionally, they can be computationally expensive, especially for large and complex models.



