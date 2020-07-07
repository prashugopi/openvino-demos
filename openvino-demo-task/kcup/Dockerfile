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
RUN pip3 install nms
RUN pip3 install statistics
RUN pip3 install pathlib

ADD demo_0418_win10 /temp/
WORKDIR /temp
ADD run.sh /temp/run.sh
RUN chmod u+x /temp/run.sh
RUN pip3 install opencv-python
CMD bash -C "/temp/run.sh"

