#!/bin/bash

cd /opt/intel/openvino_2019.1.144/bin
source setupvars.sh
cd /opt/intel/openvino_2019.1.144/deployment_tools/inference_engine/samples/build/intel64/Release 
./interactive_face_detection_demo -i cam -m /home/open_model_zoo/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml -m_ag /home/open_model_zoo/model_downloader/Retail/object_attributes/age_gender/dldt/age-gender-recognition-retail-0013.xml -m_hp /home/open_model_zoo/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.xml 
#-m_em /home/open_model_zoo/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.xml -m_lm /home/open_model_zoo/model_downloader/Transportation/object_attributes/facial_landmarks/custom-35-facial-landmarks/dldt/facial-landmarks-35-adas-0002.xml -d CPU
