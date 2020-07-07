#!/usr/bin/env python3

import cv2
import pathlib

import numpy as np
from nms.nms import boxes as nms

from openvino.inference_engine import IENetwork, IEPlugin

class KCupDetector():
    def __init__(self, model_path, cpu_exts, model_name, device,
                 detection_callback, detection_confidence=0.5):

        assert isinstance(model_path, pathlib.Path)
        assert isinstance(cpu_exts, list)

        self.detection_callback = detection_callback
        self.detection_confidence = detection_confidence
        
        self.plugin = self.load_plugin(cpu_exts, device)
        net = self.load_network(model_path, model_name)

        self.check_supported_layers(net)
        
        self.input_net_names = list(net.inputs)
        self.output_net_names = list(net.outputs)
        
        # ATTN: Detector model specific!
        # Input has shape [1, C, H, W]
        # Output has shape [1, 1, 200, 7]
        # Here, we use the 1st input from input side and 1nd output from the output side
        self.model_input_idx = 0
        self.model_output_idx = 0
        self.model_input_net_name = self.input_net_names[self.model_input_idx]
        self.model_output_net_name = self.output_net_names[self.model_output_idx]

        print('Input net name {} / Output net name {}'.format(self.model_input_net_name, self.model_output_net_name))
        
        # Compute the height and width for input/output
        self.model_input_height = net.inputs[self.model_input_net_name].shape[2]
        self.model_input_width = net.inputs[self.model_input_net_name].shape[3]
        self.model_output_height = net.outputs[self.model_output_net_name].shape[2]
        self.model_output_width = net.outputs[self.model_output_net_name].shape[3]
        
        print('Input shape {}'.format(net.inputs[self.model_input_net_name].shape))
        print('Output shape {}'.format(net.outputs[self.model_output_net_name].shape))

        self.exec_net = self.plugin.load(network=net, num_requests=1)
        
    def load_plugin(self, cpu_exts, device):
        plugin = IEPlugin(device=device)
        for cpu_ext in cpu_exts:
            assert isinstance(cpu_ext, pathlib.Path)
            plugin.add_cpu_extension(str(cpu_ext))
        return plugin

    def load_network(self, model_base, model_name):
        model_base = model_base/model_name/'FP32'/model_name
        model_xml = model_base.with_suffix('.xml')
        model_bin = model_xml.with_suffix('.bin')
        net = IENetwork(model=str(model_xml), weights=str(model_bin))
        return net
  
    def check_supported_layers(self, net):
        supported_layers = self.plugin.get_supported_layers(net)
        #print('Supported layers: {}'.format(supported_layers))
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print('Unsupported layers: {}'.format(not_supported_layers))
            raise ValueError('Unsupported layers!')

    def resize_keep_aspect(self, raw_img, shape, centering=False):
        #print('raw_img shape = {} to be resized to {}'.format(raw_img.shape, shape))
        width, height = shape
        raw_height, raw_width, _ = raw_img.shape
        aspect_ratio = raw_width / raw_height
        new_width = aspect_ratio * height
        if new_width <= width:
            scale_factor = new_width/raw_width
            img_tmp = cv2.resize(raw_img, (int(new_width), height))
        else: #if new_width > width:
            new_height = width / aspect_ratio
            scale_factor = new_height/raw_height
            img_tmp = cv2.resize(raw_img, (width, int(new_height)))
        # Fill the remaining image with black 
        delta_x = width-img_tmp.shape[1]
        delta_y = height-img_tmp.shape[0]
        if centering:
            yy = int(delta_y / 2)
            xx = int(delta_x / 2)
            img = cv2.copyMakeBorder(img_tmp, yy, yy, 
                                     xx, xx, cv2.BORDER_CONSTANT)
        else:
            img = cv2.copyMakeBorder(img_tmp, 0, delta_y, 
                                         0, delta_x, cv2.BORDER_CONSTANT)
        return img, scale_factor, delta_x, delta_y
  
    def preprocess_image(self, raw_image):
        resized_img, scale_factor, delta_x, delta_y = \
            self.resize_keep_aspect(raw_image,
                                    (self.model_input_width,
                                     self.model_input_height))
        img = resized_img.transpose((2, 0, 1)) # HWC to CHW
        img = img[np.newaxis, :, :, :]
        return img, scale_factor, delta_x, delta_y
    def update_detection_confidence(self, detection_confidence):
        self.detection_confidence = detection_confidence
        
    def detect(self, image, detect_ctx=None):
        """
        Invoke OpenVINO's inference engine in synchronous fashion

        Parameters
        ----------
        image : numpy array of image for detection. 
                We assume the image is shaped Height x Width x Channel
        detect_ctx : Extra user detection context information to return 
                     to caller in callback 

        Returns
        -------
        Invokes callback function with ndarray of bboxes
        """
        if not isinstance(image, np.ndarray):
            raise TypeError('Invalid image type {}. ndarray expected'.format(type(image)))
        assert(len(image.shape) == 3)
        bboxes = []
        scaled_image, scale_factor, delta_x, delta_y = self.preprocess_image(image)
        self.exec_net.start_async(request_id=0,
                                 inputs={self.model_input_net_name: scaled_image})
        self.exec_net.requests[0].wait(-1)
        detections = self.exec_net.requests[0].outputs[self.model_output_net_name]
        for detection_id in range(detections.shape[2]):
            _, class_id, conf, rxmin, rymin, rxmax, rymax = detections[0,0,detection_id,:]
            if conf > self.detection_confidence:
                # Note that rxmin, rymin, rxmax, rymax are all between 0 and 1
                # They are ratio of the input image dimension, so we need to
                # compute the absolute coordinate in the original image.
                xmin = int(rxmin * (delta_x / scale_factor + image.shape[1]))
                ymin = int(rymin * (delta_y / scale_factor + image.shape[0]))
                xmax = int(rxmax * (delta_x / scale_factor + image.shape[1]))
                ymax = int(rymax * (delta_y / scale_factor + image.shape[0]))
                #print('bbox = {},{}'.format((xmin, ymin), (xmax, ymax)))

                bboxes.append([xmin, ymin, xmax-xmin, ymax-ymin, class_id, conf])
        if len(bboxes) > 0:
            bboxes = np.array(bboxes)
            indices = nms(bboxes[:,:4], bboxes[:, 5])
            bboxes = list(bboxes[indices]) # bboxes has conf, so it must be float!
        else:
            bboxes = list()
        self.detection_callback(np.array(bboxes), detect_ctx)

