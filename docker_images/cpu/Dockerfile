FROM tensorflow/tensorflow:2.7.0

ENV DEBIAN_FRONTEND=noninteractive

# Install a tool to find a fast, up-to-date Ubuntu Archive Mirror.
RUN apt update && apt install --no-install-recommends -y python3-pip &&\
    pip3 install -U pip &&\
    pip3 install apt-mirror-updater &&\
    apt-mirror-updater -a -q &&\
    rm -rf /var/lib/apt/lists/*


# Install Build, OpenCV dependencies
RUN apt-mirror-updater -a -q &&\
    apt install --no-install-recommends -y build-essential cmake ninja-build git pkg-config &&\
    apt install --no-install-recommends -y libgtk2.0-dev libx11-dev libjpeg-dev libpng-dev &&\
    rm -rf /var/lib/apt/lists/*

# Install OpenCV
RUN mkdir -p /tmp/opencv &&\
    cd /tmp/opencv &&\
    git clone https://github.com/opencv/opencv.git &&\
    mkdir opencv/build &&\
    cd opencv/build &&\
    cmake -G Ninja .. \
    -DBUILD_PERF_TESTS=Off \
    -DBUILD_TESTS=Off \
    -DBUILD_EXAMPLES=Off \
    -DBUILD_opencv_apps=Off \
    -DBUILD_opencv_calib3d=On \
    -DBUILD_opencv_core=On \
    -DBUILD_opencv_dnn=Off \
    -DBUILD_opencv_features2d=On \
    -DBUILD_opencv_flann=On \
    -DBUILD_opencv_highgui=On \
    -DBUILD_opencv_imgcodecs=On \
    -DBUILD_opencv_imgproc=On \
    -DBUILD_opencv_java_bindings_generator=Off \
    -DBUILD_opencv_js=Off \
    -DBUILD_opencv_ml=Off \
    -DBUILD_opencv_objdetect=Off \
    -DBUILD_opencv_photo=Off \
    -DBUILD_opencv_python2=Off \
    -DBUILD_opencv_python3=On \
    -DBUILD_opencv_python_bindings_generator=Off \
    -DBUILD_opencv_python_tests=Off \
    -DBUILD_opencv_shape=Off \
    -DBUILD_opencv_stitching=Off \
    -DBUILD_opencv_superres=Off \
    -DBUILD_opencv_ts=Off \
    -DBUILD_opencv_video=On \
    -DBUILD_opencv_videoio=On \
    -DBUILD_opencv_videostab=Off \
    -DBUILD_opencv_world=Off \
    -DWITH_JASPER=Off \
    -DWITH_ITT=Off \
    -DWITH_JPEG=On \
    -DWITH_WEBP=Off \
    -DWITH_PNG=On \
    -DWITH_TIFF=Off \
    -DWITH_OPENEXR=Off \
    -DWITH_OPENCL=Off\
    -DWITH_IPP=Off \
    -DWITH_PROTOBUF=Off \
    -DWITH_GTK_2_X=On \
    -DCMAKE_BUILD_TYPE=Release &&\
    ninja &&\
    ninja install &&\
    cd ~ &&\
    rm -rf /tmp/opencv

# Boost
RUN apt-mirror-updater -a -q &&\
    apt install --no-install-recommends -y libboost-python-dev libboost-numpy-dev &&\
    rm -rf /var/lib/apt/lists/*

# OpenCV (Python)
RUN pip3 install -U pip &&\
    pip3 install opencv-python

# Yaml
RUN pip3 install pyyaml

# LabelImg
RUN apt-mirror-updater -a -q &&\
    apt install --no-install-recommends -y pyqt5-dev-tools &&\
    pip3 install -U pip &&\
    pip3 install labelImg

# Model conversion
RUN pip3 install -U pip &&\
    pip3 install tf2onnx

RUN ldconfig
