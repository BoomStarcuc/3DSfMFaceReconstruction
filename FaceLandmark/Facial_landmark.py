import torch
import torch.nn as nn
import torch.optim as optim
from models import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

net = ResNet18(1756).to(device)
net.load_state_dict(torch.load('./model/model_keypoints_800pts_iter_final.pt'))
net.eval()

image1 = mpimg.imread('./test_landmark/F1001_A_0000_090.png')
if(image1.shape[2] == 4):
  image1 = image1[:,:,0:3]
image1 = np.copy(image1)/255.0

image1 = torch.from_numpy(image1.transpose((2, 0, 1))).unsqueeze(0)


image2 = mpimg.imread('./test_landmark/F1001_A_0000_030.png')
if(image2.shape[2] == 4):
  image2 = image2[:,:,0:3]
image2 = np.copy(image2)/255.0

image2 = torch.from_numpy(image2.transpose((2, 0, 1))).unsqueeze(0)


if (torch.cuda.is_available()):
    image1 = image1.type(torch.cuda.FloatTensor)
    image2 = image2.type(torch.cuda.FloatTensor)
    image1.to(device)
    image2.to(device)
else:
    image1 = image1.type(torch.FloatTensor)
    image2 = image2.type(torch.FloatTensor)

output1 = net(image1)
output1 = output1.view(output1.size()[0], -1, 2)
output_pts1 = output1[:,:,:2]

output2 = net(image2)
output2 = output2.view(output2.size()[0], -1, 2)
output_pts2 = output2[:,:,:2]

# un-transform the image data
image1 = image1.data   # get the image from it's Variable wrapper
image2 = image2.data
if (torch.cuda.is_available()):
    image1 = image1.cpu()
    image2 = image2.cpu()
    
image1 = image1.numpy()   # convert to numpy array from a Tensor
image1 = np.transpose(image1[0], (1, 2, 0))   # transpose to go from torch to numpy image

image2 = image2.numpy()   # convert to numpy array from a Tensor
image2 = np.transpose(image2[0], (1, 2, 0))   # transpose to go from torch to numpy image

#un-transform the predicted key_pts data
predicted_key_pts1 = output_pts1[0].data
predicted_key_pts2 = output_pts2[0].data

if (torch.cuda.is_available()):
    predicted_key_pts1 = predicted_key_pts1.cpu()
    predicted_key_pts2 = predicted_key_pts2.cpu()

predicted_key_pts1 = predicted_key_pts1.numpy()
predicted_key_pts2 = predicted_key_pts2.numpy()

#predicted_key_pts = predicted_key_pts*mask
#undo normalization of keypoints  
predicted_key_pts1 = predicted_key_pts1*50+100
predicted_key_pts2 = predicted_key_pts2*50+100
    
outimg = np.hstack((image1*255.0,image2*255.0))

plt.figure()
plt.scatter(predicted_key_pts1[::14,0],predicted_key_pts1[::14,1],s=1,color='green')
plt.scatter(predicted_key_pts2[::14,0]+384,predicted_key_pts2[::14,1],s=1,color='orange')
plt.plot([predicted_key_pts1[::14,0],predicted_key_pts2[::14,0]+384],[predicted_key_pts1[::14,1],predicted_key_pts2[::14,1]])
plt.imshow(outimg)
plt.axis('off')

plt.show()