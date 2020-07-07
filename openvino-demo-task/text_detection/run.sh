#!/bin/bash

cd /opt/intel/openvino_2019.1.144/bin
source setupvars.sh
cd /opt/intel/openvino_2019.1.144/deployment_tools/inference_engine/samples/build/intel64/Release 
./text_detection_demo -i /dev/video0 -dt webcam -m_tr /home/open_model_zoo/model_downloader/Retail/text_recognition/bilstm_crnn_bilstm_decoder/0012/dldt/text-recognition-0012.xml -m_td /home/open_model_zoo/model_downloader/Retail/object_detection/text/pixel_link_mobilenet_v2/0001/text-detection-0002.xml
