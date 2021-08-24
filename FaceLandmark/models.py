import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
#import torch.nn.init as I

class Net_(nn.Module):

    def __init__(self):
        super(Net_, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 4x4 square convolution kernel
        # output size = (W-F)/S +1 = (96-4)/1 +1 = 93
        # the output Tensor for one image, will have the dimensions: (32, 93, 93)
        # after one pool layer, this becomes (32, 46, 46)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 46, 3)
        self.conv3 = nn.Conv2d(46, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fcl1 = nn.Linear(6400, 1000)
        self.fcl2 = nn.Linear(1000, 500)
        self.fcl3 = nn.Linear(500, 136)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
    def forward(self, x):
        ## Implementation of NaimishNet as in https://arxiv.org/pdf/1710.00977.pdf
        
        x = self.conv1(x)           # 2 - Convolution2d1
        x = F.elu(x)                # 3 - Activation1
        x = self.pool(x)            # 4 - Maxpooling2d1     
        x = self.dropout1(x)        # 5 - Dropout1      
        x = self.conv2(x)           # 6 - Convolution2d2
        x = F.elu(x)                # 7 - Activation2
        x = self.pool(x)            # 8 - Maxpooling2d2
        x = self.dropout2(x)        # 9 - Dropout2
        x = self.conv3(x)           # 10 - Convolution2d3
        x = F.elu(x)                # 11 - Activation3
        x = self.pool(x)            # 12 - Maxpooling2d3
        x = self.dropout3(x)        # 13 - Dropout3
        x = self.conv4(x)           # 14 - Convolution2d4
        x = F.elu(x)                # 15 - Activation4
        x = self.pool(x)            # 16 - Maxpooling2d4
        x = self.dropout4(x)        # 17 - Dropout4
        x = x.view(x.size(0), -1)   # 18 - Flatten1
        x = self.fcl1(x)            # 19 - Dense1
        x = F.elu(x)                # 20 - Activation5
        x = self.dropout5(x)        # 21 - Dropout5
        x = self.fcl2(x)            # 22 - Dense2
        x = F.elu(x)                # 23 - Activation6
        x = self.dropout6(x)        # 24 - Dropout6
        x = self.fcl3(x)            # 25 - Dense3 

        # a modified x, having gone through all the layers of your model, should be returned
        return x


class Net(nn.Module):

    def __init__(self,num_classes):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        self.conv1 = nn.Conv2d(3, 32, 7, 3, 1)  ## output size = (W-F+2P)/S +1 = (224-7+2)/3 +1 = 74
        self.conv2 = nn.Conv2d(32, 64, 5, 3, 0)  ## output size = (W-F+2P)/S +1 = (74-5)/3 +1 = 24
        self.conv3 = nn.Conv2d(64, 128, 5, 3, 1)  ## output size = (24-5+2)/3 +1 = 8
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 0)  ## output size = (8-3)/1 +1 = 6
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 0)  ## output size = (6-3)/1 +1 = 4
        self.conv6 = nn.Conv2d(512, 512, 1, 1, 0)  ## output size = (4-1)/1 +1 = 4

        self.fc1 = nn.Linear(4 * 4 * 512, 1024)
        self.fc1_drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_drop = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # Apply convolutional layers
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6(x))

        # Flatten and continue with dense layers
        x = x.view(x.size(0), -1)
        x = F.selu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.selu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x


from torchvision import models
class ResNet18(nn.Module):
    def __init__(self,num_classes):
        super(ResNet18, self).__init__()
        self.model_name='resnet18'
        self.model=models.resnet18(pretrained=True)
        self.model.fc=nn.Linear(self.model.fc.in_features,num_classes)

    def forward(self, x):
        x=self.model(x)
        return x

class SqueezeNet(nn.Module):
    def __init__(self,num_classes):
        super(SqueezeNet,self).__init__()
        self.pretrain_net = models.squeezenet1_1(pretrained=True)
        self.base_net = self.pretrain_net.features
        self.pooling  = nn.AvgPool2d(13)
        self.fc = nn.Linear(512,num_classes)
    def forward(self,x):
        x = self.base_net(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
