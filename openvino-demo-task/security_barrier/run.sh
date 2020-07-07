#!/bin/bash

cd /opt/intel/openvino_2019.1.144/bin
source setupvars.sh
cd /opt/intel/openvino_2019.1.144/deployment_tools/inference_engine/samples/build/intel64/Release 
./security_barrier_camera_demo -i cam -m /home/open_model_zoo/model_downloader/Security/object_detection/barrier/0106/dldt/vehicle-license-plate-detection-barrier-0106.xml -m_va /home/open_model_zoo/model_downloader/Security/object_attributes/vehicle/resnet10_update_1/dldt/vehicle-attributes-recognition-barrier-0039.xml -m_lpr /home/open_model_zoo/model_downloader/Security/optical_character_recognition/license_plate/dldt/license-plate-recognition-barrier-0001.xml -d CPU
