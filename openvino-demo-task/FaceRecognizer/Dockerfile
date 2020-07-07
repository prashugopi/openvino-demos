FROM ubuntu:16.04
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY

ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/15512/l_openvino_toolkit_p_2019.1.144.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    python\
    python-pip\
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR

RUN $INSTALL_DIR/install_dependencies/install_openvino_dependencies.sh

# build Inference Engine samples
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"

RUN apt-get install -y python-setuptools python-opencv


WORKDIR /home
RUN apt install python3-pip -y
RUN pip3 install requests
RUN apt install -y git && \
git clone https://github.com/opencv/open_model_zoo.git
WORKDIR /home/open_model_zoo/model_downloader
RUN git checkout 2019_R1
RUN pip3 install pyyaml
RUN python3 downloader.py --name face-detection-adas-0001
RUN python3 downloader.py --name age-gender-recognition-retail-0013
RUN python3 downloader.py --name head-pose-estimation-adas-0001
RUN python3 downloader.py --name emotions-recognition-retail-0003
RUN python3 downloader.py --name facial-landmarks-35-adas-0002
RUN apt install -y libcanberra-gtk-module libcanberra-gtk3-module
ADD run.sh /home/run.sh
RUN chmod u+x /home/run.sh
CMD bash -C "/home/run.sh"

# docker run -it --rm --privileged -v /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=${DISPLAY} face_recognizer
