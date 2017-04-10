
# AI in Content Moderation
## RESTful Web Service and C++ compilable version of [YOLO](https://arxiv.org/abs/1612.08242) written in C and CUDA for object detection.

## Features

* Uses same code-base as original yolo (ie same .c files are used). Modifications include runtime bug-fixes, compile time fixes for c++, the build system itself, and wrapping a REST API around it.

* Build system supports 2 targets - 
  * original darknet-yolo (with gcc compiler), 
  * darknet-cpp (with g++ compiler)

* Up and running with OpenCV3, Ubuntu 16.04 and CUDA 8.x

## Usage

 * `make darknet` - only yolo (original code), with OPENCV=0
 * `make darknet-cpp` - only the CPP version, with OPENCV=1

## External Dependencies

### OpenCV

  OpenCV was designed for computational efficiency and with a strong focus on real-time applications. Written in optimized C/C++, the library can take advantage of multi-core processing.

  Download the latest source archive for OpenCV 3.1 from https://github.com/opencv/opencv. (Do not download it from http://opencv.org/downloads.html, because the official OpenCV 3.1 does not support CUDA 8.0.)

  **Compilation and Installation**

  ```
  git clone https://github.com/opencv/opencv
  mkdir opencv/build
  cd opencv/build
  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..    
  cd opencv/build
  make -j $(($(nproc) + 1))
  make install
  ```

### Jansson
	
  Jansson is a C library for encoding, decoding and manipulating JSON data.

  Jansson is available on Jansson's official github! [here](https://github.com/akheron/jansson)

  **Compilation and Installation**

  ```
  git clone https://github.com/akheron/jansson.git
  autoreconf -fi
  ./configure
  make
  make check
  make install
  ```

### Restbed

  Restbed is a comprehensive and consistent programming model for building applications that require seamless and secure communication over HTTP, with the ability to model a range of business processes, designed to target mobile, tablet, desktop and embedded production environments.

  Restbed is available on Restbed's official github! [here](https://github.com/Corvusoft/restbed)

  **Compilation and Installation**

  ```
  cd /usr/local/lib
  git clone --recursive https://github.com/corvusoft/restbed.git
  mkdir /usr/local/lib/restbed/build
  cd /usr/local/lib/restbed/build
  cmake [-DBUILD_TESTS=YES] [-DBUILD_EXAMPLES=YES] [-DBUILD_SSL=NO] [-DBUILD_SHARED=YES] [-DCMAKE_INSTALL_PREFIX=/output-directory] ..
  cd /usr/local/lib/restbed/build
  make -j $(($(nproc) + 1)) install
  ```

## Train as below
	
  To train the model you will need all of the VOC data from 2007 to 2012 (or different datasets). You can find links to the data [here](https://pjreddie.com/projects/pascal-voc-dataset-mirror/). 

  * Now we need to generate the label files. We need a .txt file for each image with a line for each ground truth object in the image that looks like:

  `<object-class> <x> <y> <width> <height>`

  * To generate these file we will run the voc_label.py script in scripts/ directory.

  ```
  python voc_label.py
  ```

  * Now go to your project directory. Change the files per below:

  ```
  yolo-voc.cfg - change line classes=20 to suit desired number of classes
  yolo-voc.cfg - change the number of filters in the CONV layer above the region layer - (#classes + 4 + 1)*(5)
  voc.data - change line classes=20, and paths to training image list file
  voc.names - number of lines must be equal the number of classes
  ```

  * For training we use convolutional weights that are pre-trained on Imagenet. You can just download the weights for the convolutional layers [here](http://pjreddie.com/media/files/darknet19_448.conv.23).

  * Now we can train! Run the command:

  `./darknet-cpp detector train ./cfg/voc-myclasses.data ./cfg/yolo-myconfig.cfg darknet19_448.conv.23`

## Run the server

  * Now we can run the object detection server! Run the command:

  `./darknet-cpp detector load ./cfg/voc-myclasses.data ./cfg/yolo-myconfig.cfg my-yolo.weights`

## How to deploy on Amazon EC2 with Docker (GPU support)

  * Assuming the NVIDIA drivers and Docker are properly installed. Install nvidia-docker and nvidia-docker-plugin. 

  ```  
  wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
  sudo dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
  ```
  * Now go to your project directory. In DockerImage/ folder run the command:

  `docker build -t <image-name> .`

  * Wait for the image to be built and run a container of that image by doing:

  `nvidia-docker run -it -p 1984:1984 <image-name> /bin/bash`

  * Just run the server inside the container and have fun!

![ai-content-moderation](http://images.memes.com/meme/893073)





