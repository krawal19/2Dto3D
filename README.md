# 2D to 3D Reconstruction*
## This repository contains the collection of methods for doing 3D reconstrcution from 2D stereo images.
#### * Repository work is currently under development and will be updated soon.

## Dependencies
Follow the link below for installation of opencv with CMake and GCC

- [OpenCV](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html) 3.3.1-dev or later
- GCC 4.4.x or later
- CMake 2.8.7 or higher
- Git
- pkg-config
- stereo images with calibration matrix or stereo camera

## Building the code
Follow the steps below to build and run the code
```
git clone https://github.com/krawal19/2Dto3D.git
cd 2Dto3D
mkdir build
cd build 
cmake ..
make
```
This will build the code. Now to Run follow the steps below
```
cd /2Dto3D/build
./stereo_disparity
./3D_reconstruction
```
Output of the above program will be stored in the output folder.

## References
- 3D reconstruction with vision medium blog [https://towardsdatascience.com/3-d-reconstruction-with-vision-ef0f80cbb299]

- Iterative linear triangulation blog [https://www.morethantechnical.com/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/]
- Multiple View Geometry in Computer Vision, Hartley, R. I. and Zisserman, A., 2004, Cambridge University Press [http://www.robots.ox.ac.uk/~vgg/hzbook/]