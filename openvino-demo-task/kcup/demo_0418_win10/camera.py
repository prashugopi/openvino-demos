import time

#import numpy as np
import cv2

class USBCamera():
    def __init__(self, stream=0, height=720, width=1280):
        self.fps = 1
        self.height = height
        self.width = width
        self.cam = None
        self.stream = stream

    def read_iter(self):
        if self.cam is None:
            raise ValueError('Call start() first!')
        while True:
            ret, frame = self.cam.read()
            if not ret:
                time.sleep(1/self.fps)
                continue
            yield frame
    
    def get_properties(self):
        assert (self.cam is not None)
        auto = self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        exp = self.cam.get(cv2.CAP_PROP_EXPOSURE)
        con = self.cam.get(cv2.CAP_PROP_CONTRAST)
        bri = self.cam.get(cv2.CAP_PROP_BRIGHTNESS)
        return auto, exp, con, bri

    def get_property(self, key):
        assert (self.cam is not None)
        if key == 'auto':
            return self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        if key == 'exposure':
            return self.cam.get(cv2.CAP_PROP_EXPOSURE)
        elif key == 'contrast':
            return self.cam.get(cv2.CAP_PROP_CONTRAST)
        elif key == 'brightness':
            return self.cam.get(cv2.CAP_PROP_BRIGHTNESS)

    def set_property(self, key, value):
        assert (self.cam is not None)
        if key == 'auto':
            curr = self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0 if curr == 1 else 1)
        if key == 'exposure':
            self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.cam.set(cv2.CAP_PROP_EXPOSURE, value)
        elif key == 'contrast':
            self.cam.set(cv2.CAP_PROP_CONTRAST, value)
        elif key == 'brightness':
            self.cam.set(cv2.CAP_PROP_BRIGHTNESS, value)

    def start(self, auto_focus=True):
        print('Opening video stream {}'.format(self.stream))
        self.cam = cv2.VideoCapture(self.stream)
        if not self.cam.isOpened():
            raise ValueError('Failed to open camera {}'.format(self.stream))
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1 if auto_focus else 0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)
        print('Starting camera with {}x{} @ {}fps'.format(self.width, self.height, self.fps))
        #print('FourCC = {}'.format(self.cam.get(cv2.CAP_PROP_FOURCC)))
    def stop(self):
        self.cam.release()

class FileCamera():
    def __init__(self, filename):
        self.filename = filename
        self.cam = cv2.VideoCapture(filename)
        if not self.cam.isOpened():
            raise ValueError('Failed to open video file {}'.format(filename))
        self.fps = self.cam.get(cv2.CAP_PROP_FPS)
        self.width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.cam.release()
        self.cam = None

    def start(self):
        self.cam = cv2.VideoCapture(self.filename)
        if not self.cam.isOpened():
          raise ValueError('Failed to open video file {}'.format(filename))

    def stop(self):
        self.cam.release()
        
    def read_iter(self):
        if self.cam is None:
            raise ValueError('Call start() first!')
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            yield frame
            time.sleep(1/self.fps)

        
