#!/bin/bash

cd /opt/intel/openvino_2019.1.144/bin
source setupvars.sh
cd /opt/intel/openvino_2019.1.144/deployment_tools/inference_engine/samples/build/intel64/Release 
./crossroad_camera_demo -i cam -m /home/open_model_zoo/model_downloader/Security/object_detection/crossroad/0078/dldt/person-vehicle-bike-detection-crossroad-0078.xml -m_pa /home/open_model_zoo/model_downloader/Security/object_attributes/pedestrian/person-attributes-recognition-crossroad-0230/dldt/person-attributes-recognition-crossroad-0230.xml -m_reid /home/open_model_zoo/model_downloader/Retail/object_reidentification/pedestrian/rmnet_based/0079/dldt/person-reidentification-retail-0079.xml -d CPU
