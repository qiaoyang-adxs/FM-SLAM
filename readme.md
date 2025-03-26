# FM-SLAM

## Installation

### 1、Download this repo

```
git clone git@github.com:qiaoyang-adxs/FM-SLAM.git
```

### 2、Environment

install ORB-SLAM2 dependencies

```
sudo apt update

sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev libjasper-dev

sudo apt-get install libglew-dev libboost-all-dev libssl-dev
```

OpenCV

```
~$ git clone https://github.com/opencv/opencv.git
~$ git clone https://github.com/opencv/opencv_contrib.git
~$ cd opencv
~/opencv$ git checkout 4.2.0
~/opencv$ cd ../opencv_contrib
~/opencv_contrib$ git checkout 4.2.0
~/opencv_contrib$ cd ../opencv
~/opencv$ mkdir build && cd build
~/opencv/build$ cmake -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
~/opencv/build$ make -j8
~/opencv/build$ sudo make install
```

Eigen3

```
sudo apt install libeigen3-dev
```

Pangolin

```
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin 
git checkout v0.6
mkdir build 
cd build 
cmake .. -D CMAKE_BUILD_TYPE=Release 
make -j4
sudo make install
```

3、install FM-SLAM

```
chmod +x build.sh
./build.sh
```

# Quick Start

Modify the path according to the dataset

```
 ./Examples/RGB-D/rgbd_tum_masked Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml PATH/TO/Dataset/rgbd_dataset_freiburg3_walking_xyz PATH/TO/Dataset/rgbd_dataset_freiburg3_walking_xyz/associations_modified.txt
```

