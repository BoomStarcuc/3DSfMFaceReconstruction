# (ACMMM 2021 oral) SfM Face Reconstruction Based on Massive Landmark
This repository shows two tasks: Face landmark detection and Face 3D reconstruction, which is described in this paper: Deep Unsupervised 3D SfM Face Reconstruction Based on Massive Landmark Bundle Adjustment.

## Installation
1. Clone the repository.
2. install dependencies.

```
pip install -r requirement.txt
```

# Face landmark detection
<div align=center><img src="https://github.com/BoomStarcuc/3DSfMFaceReconstruction/blob/master/data/RedAndGreen.png" width="375" height="265"/><img src="https://github.com/BoomStarcuc/3DSfMFaceReconstruction/blob/master/data/Picture1_crop.jpg" width="375" height="265"/></div>

## Running a pre-trained model
1. Download landmark pre-trained model at [GoogleDrive](https://drive.google.com/file/d/1tDqX2nG1qATqrd2fEb4Sgs4av25d9tgN/view?usp=sharing), and put it into ```FaceLandmark/model/```
2. Run the test file

```
python Facial_landmark.py
```


# Face 3D reconstruction
<div align=center><img src="https://github.com/BoomStarcuc/3DSfMFaceReconstruction/blob/master/data/Stirling ESRC 3D.png" width="410" height="265"/><img src="https://github.com/BoomStarcuc/3DSfMFaceReconstruction/blob/master/data/Facescape%20face.png" width="410" height="265"/></div>

## Running a pre-trained model
1. Download face 3D reconstruction pre-trained model at [GoogleDrive](https://drive.google.com/file/d/1t-3IXQHn5DmXpoumf5a8JfQgWxg54krW/view?usp=sharing), and put it into ```FaceReconstruction/checkpoints/```

3. Run the ```inference.py``` file to generate disparity map

```
python inference.py --dataset-dir './FaceReconstruction/test_image/' --output-dir './FaceReconstruction/output/' --pretrained './FaceReconstruction/checkpoints/dispnet_model_best.pth.tar' --resnet-layers 18 --output-disp 
```
4. Run the ```generate_ply.py``` file to generate point cloud ```.ply``` file

```
python generate_ply.py
```
