FROM tensorflow/tensorflow:latest-gpu
RUN add-apt-repository "deb http://us.archive.ubuntu.com/ubuntu trusty main multiverse"
RUN apt-get update
RUN apt-get install --assume-yes build-essential cmake git \
	build-essential pkg-config unzip ffmpeg qtbase5-dev python-dev python3-dev python-numpy python3-numpy\
	libopencv-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev\
	libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev\
	libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev\
	libvorbis-dev libxvidcore-dev v4l-utils\
	&& apt-get clean

# Build OpenCV
RUN git clone https://github.com/opencv/opencv
RUN mkdir opencv/build
RUN cd opencv/build ; cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..    
RUN cd opencv/build && make -j $(($(nproc) + 1)) && make install && cd ../.. && rm -R -f opencv

# Install OpenCV
# RUN cd opencv/build ; make install
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf
RUN ldconfig

RUN apt-get install --assume-yes openssl
RUN apt-get install --assume-yes libssl-dev

# Install restbed
RUN cd /usr/local/lib ; git clone --recursive https://github.com/corvusoft/restbed.git
RUN mkdir /usr/local/lib/restbed/build
RUN cd /usr/local/lib/restbed/build ; cmake [-DBUILD_TESTS=YES] [-DBUILD_EXAMPLES=YES] [-DBUILD_SSL=NO] [-DBUILD_SHARED=YES] [-DCMAKE_INSTALL_PREFIX=/output-directory] ..
RUN cd /usr/local/lib/restbed/build ; make -j $(($(nproc) + 1)) install

RUN apt-get install --assume-yes autotools-dev
RUN apt-get install --assume-yes autoconf
RUN apt-get install --assume-yes automake
RUN apt-get install --assume-yes libtool-bin

# Install jansson
RUN git clone https://github.com/akheron/jansson.git
RUN cd jansson ; autoreconf -fi
RUN cd jansson ; ./configure
RUN cd jansson ; make
RUN cd jansson ; make check
RUN cd jansson ; make install

# Install App
RUN mkdir -p /root/.ssh
# Copy over private key, and set permissions - remember doing ssh-keygen -t rsa -C EID@innersource.accenture.com if there's no shh key
COPY id_rsa /root/.ssh
RUN chmod 600 /root/.ssh/id_rsa
# Add bitbuckets key to known_hosts
RUN ssh-keyscan innersource.accenture.com >> /root/.ssh/known_hosts
RUN git clone ssh://git@innersource.accenture.com/cfi/aicm-ai-datamodels.git
RUN echo “/usr/local/lib” > /etc/ld.so.conf.d/usr_local.conf
RUN /sbin/ldconfig
RUN cd aicm-ai-datamodels/object-detection ; make darknet-cpp