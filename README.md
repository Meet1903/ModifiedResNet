# ModifiedResNet
This mini-project involved designing a modified ResNet architecture for image classification on the CIFAR-10 dataset, with the constraint that the model must have no more than 5 million trainable parameters. The ResNet architecture was implemented with skip connections and residual blocks, and hyperparameters such as the number of channels, filter size, kernel size, and pool size were adjusted to optimize model accuracy. Various optimization techniques, data augmentation strategies, regularizers, and choice of the learning rate, batch size, and epochs were experimented with to further improve model performance. The final model was trained from scratch and achieved high accuracy on the CIFAR-10 test set within the constraints of the project. 

# Methodology
ModifiedResNet builds upon the ResNet architecture by making several modifications to improve its performance. The architecture consists of several layers of convolutional neural networks, with each layer having a number of blocks. The blocks themselves are made up of convolutional layers followed by batch normalization layers and a residual connection. The residual connection allows the network to learn the difference between the output of a layer and its input, which helps to reduce the vanishing gradient problem and allows the network to be trained much deeper than traditional networks.

The first layer of the model is a convolutional layer with 64 filters and a kernel size of 3x3. This is followed by a batch normalization layer and a ReLU activation function. The output of this layer is then passed through a series of three layers, each containing several blocks of convolutional layers with batch normalization and ReLU activation functions, and a residual connection. The number of blocks in each layer is specified by the num_blocks parameter passed to the constructor.

The first layer contains 2 blocks, the second layer contains 2 blocks with a stride of 2, and the third layer contains 3 blocks with a stride of 2 [model 3]. The stride of 2 reduces the spatial resolution of the feature maps, which allows the network to learn higher-level features that are more invariant to small changes in the input image. The dropout layer is added after the second layer to prevent overfitting.

After the final layer of blocks, the output is passed through an average pooling layer with a kernel size of 8x8, which reduces the size of the feature maps to 1x1. Finally, the output is flattened and passed through a fully connected layer with 10 output nodes, corresponding to the 10 classes in the CIFAR-10 dataset.

We have experimented with multiple models, altering various configurations such as the number of stages, the optimizer used, and the number of blocks employed. Each model was trained on 60 epochs with a batch size of 32 for both train and test data. Convolutional channel of 64, 128, and 256 is used in all the model. We have provided a visual representation of the training accuracy and loss values for each model in a graph that is included in this report.

# Result
Model 4 appears to be a viable option for image classification tasks based on its test accuracy of 90.58%, and its number of trainable parameters is 4,024,394. Additionally, the use of SGD optimizer has shown better performance over ADAM optimizer in this particular model, as suggested by the paper (Smith et al.).

Architecture of Model 4: <br />
The convolutional layer 1 has 3 input channels and 64 output channels, with a kernel size of 3 and a stride of 1. The normalization layer performs batch normalization on the output of the convolutional layer 1.

Layer 1 consists of 2 BasicBlock modules, each with 64 input channels and 64 output channels. The output of layer 1 is then passed through layer 2, which consists of 2 BasicBlock modules. The same pattern continues for layer 3 and layer 4 each with two and one BasicBlock respectively. The out_channels of the final convolutional layer in layer 4 are set to 256, and the output of the avg_pool layer will also have 256 channels with a kernel size of 4.

The next layer performs global average pooling on the output of layer 4. The Dropout layer [probability = 0.5] randomly drops out some of the values in this tensor to prevent overfitting. The linear layer takes the output of the Dropout layer and maps it to the number of classes (10). 

Furthermore, the results suggest that Model 3, which has three layers with each stage containing 2, 2, and 2 blocks, respectively, performs better than Model 1 and Model 2. This implies that the architecture of the ResNet model plays an important role in achieving high accuracy on image classification tasks.

Overall, the results are promising and provide insights into the importance of selecting an appropriate optimizer and architecture for image classification tasks.

Through experimentation with different optimization techniques, regularizers, and choice of the learning rate, batch size, and epochs, we were able to achieve high accuracy on the CIFAR-10 test set within the constraints of the project. Lessons learned from this design process include the importance of balancing the trade-off between model complexity and generalization ability, and the effectiveness of data augmentation in improving model performance among others.
