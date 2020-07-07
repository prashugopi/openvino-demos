#!/usr/bin/env python3

import time
from datetime import datetime
import random
import pathlib
import logging
import collections
import statistics

import cv2

from camera import USBCamera
from kcup_detector import KCupDetector

logging.basicConfig()
    
class App():
    def __init__(self, model_path, cpu_exts, model_name, device,
        detection_confidence, image_path, image_display_time,
        smoothing_windown_size, freq_threshold, image_shape,
        white_list, logging_level):

        self.detection_confidence = detection_confidence
        self.detector = KCupDetector(model_path=model_path,
                    cpu_exts=cpu_exts, model_name=model_name, device=device,
                    detection_callback=self.detection_callback,
                    detection_confidence=detection_confidence)

        self.logger = logging.getLogger('App')
        self.logger.setLevel(logging_level)
        self.image_display_time = image_display_time
        self.image_display_end = time.time()

        self.history = collections.deque(maxlen=smoothing_windown_size)
        self.freq_threshold =freq_threshold

        w, h = image_shape
        self.images = self.load_images(image_path, "0*.jpg", w, h)
        self.bg_images = self.load_images(image_path, "background*.jpg", w, h)

        # First background image
        self.bg_image_idx = 0

        # A white list of what SKU to be reported
        self.white_list = white_list

        # For full screen mode..
        #cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty('Display', cv2.WND_PROP_FULLSCREEN, 1)

        cv2.namedWindow('Display')
        #cv2.setWindowProperty('Display', cv2.WND_PROP_FULLSCREEN, 0)

        cv2.namedWindow('Live')
        #cv2.setWindowProperty('Live', cv2.WND_PROP_FULLSCREEN, 0)

    def load_images(self, image_path, pattern, w, h):
        # Load images with pattern 0xx.jpg into memory 
        # for corresponding package image with class id xx
        ret = []
        g = image_path.glob(pattern)
        filenames = sorted([x for x in g])
        for filename in filenames:
            print('Loading {}...'.format(str(filename)))
            img = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
            img, _, _, _ = self.detector.resize_keep_aspect(img, (w,h), centering=True)
            ret.append(img)
        return ret

    def draw_live_view(self, frame, bboxes, down_counter):
        """
        Display camera view on the live monitor
        """
        if len(bboxes) >= 1:
            xmin, ymin, w, h, class_id, conf = bboxes[0]
            xmin, ymin, w, h, class_id, conf = int(xmin), int(ymin), int(w), int(h), int(class_id), int(round(conf*100, 1))
            frame = cv2.rectangle(frame, (xmin, ymin), (xmin+w, ymin+h), (255,255,0), 2)
            frame = cv2.putText(frame, '{} {}%'.format(class_id, conf), (xmin-10, ymin),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,0))
        cv2.imshow('Live', frame)

    def smooth(self, bboxes):
        """
        Implements a sliding window smoother to find out
        the most frequent SKU and its frequency.
        Returns
           The most frequent SKU if it's above the threshold
           0 otherwise
        """
        if len(bboxes) == 0:
            self.history.append(0)
        else:
             _, _, _, _, class_id, conf = bboxes[0]
             self.history.append(int(class_id))

        if len(self.history) < self.history.maxlen:
            return 0

        try:
            # Find the most frequent SKU and frequency
            most_freq = statistics.mode(self.history)
            most_freq_count = self.history.count(most_freq)
            freq = most_freq_count / self.history.maxlen
            if  freq > self.freq_threshold:
                return most_freq
        except statistics.StatisticsError:
            pass
        return 0

    def display_image(self, class_id):
        """
        Display images on the customer monitor
        """
        if class_id >= len(self.images):
            self.logger.error('No image for class {}!'.format(class_id))
        elif class_id == 0:
            cv2.imshow('Display', self.bg_images[self.bg_image_idx])
        else:
            cv2.imshow('Display', self.images[class_id])

    def filter_bboxes(self, bboxes):

        def filter_by_class_id(bbox):
             _, _, _, _, class_id, _ = bbox
             ret = int(class_id) in self.white_list
             return ret
        return list(filter(filter_by_class_id, bboxes))

    def detection_callback(self, bboxes, detect_ctx):
        """
        Callback function. Invoked after inference is completed.
        """
        frame = detect_ctx
        time_now = time.time()
        bboxes = self.filter_bboxes(bboxes)
        self.draw_live_view(frame, bboxes, max(int(self.image_display_end-time_now),0))
        smoothed_class_id = self.smooth(bboxes)



        #  Timer expired | ID == 0  ||   What to do
        #       N        |    N     ||   Don't update screen       
        #       N        |    Y     ||   Don't update screen
        #       Y        |    N     ||   Update screen, start timer
        #       Y        |    Y     ||   Draw background, don't start timer

        if time_now > self.image_display_end:
            if smoothed_class_id == 0:
                self.display_image(0)
            else:
                self.image_display_end = time_now + self.image_display_time
                self.bg_image_idx = random.randrange(len(self.bg_images))
                self.display_image(smoothed_class_id)
                if smoothed_class_id != 0:
                    t = datetime.today()
                    row = '{},{},{},{},{},{},{}\n'.format(t.year, t.month, t.day, t.hour, t.minute, t.second, smoothed_class_id)
                    filename = 'kcup_log_{}{:02}.csv'.format(t.year, t.month)
                    with open(filename, "a+") as csvfile:
                        csvfile.write(row)
    
    def camera_property_increase(self, video, property, step=1.0):
        curr_value = video.get_property(property)
        video.set_property(property, curr_value + step)
        return video.get_property(property)

    def camera_property_decrease(self, video, property, step=1.0):
        curr_value = video.get_property(property)
        video.set_property(property, curr_value - step)
        return video.get_property(property)

    def run_from_video(self, video):
        video_iter = video.read_iter()
        auto, exp, con, bri = video.get_properties()
        while True:
            try:
                frame = next(video_iter)
                self.detector.detect(frame, detect_ctx=frame)
                # Capture keystroke
                # q - exit program
                # a - increase detection confidence filtering by 1%
                # z - decrease detection confidence filtering by 1%
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') and self.detection_confidence < 1.0:
                    self.detection_confidence = self.detection_confidence + 0.01
                    self.detector.update_detection_confidence(self.detection_confidence)
                    print('Setting detection confidence to {}'.format(self.detection_confidence))
                if k == ord('a') and self.detection_confidence > 0.0:
                    self.detection_confidence = self.detection_confidence - 0.01
                    self.detector.update_detection_confidence(self.detection_confidence)
                    print('Setting detection confidence to {}'.format(self.detection_confidence))
                if k == ord('e'):
                    print('Exposure: {}'.format(self.camera_property_increase(video, 'exposure')))
                if k == ord('d'):
                    print('Exposure: {}'.format(self.camera_property_decrease(video, 'exposure')))
                if k == ord('r'):
                    print('Contrast: {}'.format(self.camera_property_increase(video, 'contrast')))
                if k == ord('f'):
                    print('Contrast: {}'.format(self.camera_property_decrease(video, 'contrast')))
                if k == ord('t'):
                    print('Brightness: {}'.format(self.camera_property_increase(video, 'brightness')))
                if k == ord('g'):
                    print('Brightness: {}'.format(self.camera_property_decrease(video, 'brightness')))
                if k == ord('w'):
                    print('AutoExposure: {}'.format(self.camera_property_increase(video, 'auto')))
                if k == ord('h'):
                    print('Detection confidence: q / a')
                    print('AutoExposure toggle : w')
                    print('Exposure            : e / d')
                    print('Contrast            : r / f')
                    print('Brightness          : t / g')
                    print('ESC to quit')

                if k == 27: 
                        break
            except StopIteration:
                break
        self.logger.debug('Done handling all frames')
        cv2.destroyAllWindows()

if __name__ == '__main__':

    # Location of model file
    # Here, we will use ./models/RSPA/FP32/RSPA.xml 
    #                   ./models/RSPA/FP32/RSPA.bin
    model_name = 'RSPA'
    model_path = pathlib.Path(r"./models/")

    # CPU Extension locations
    # - Alex's VM
    #cpu_exts = [pathlib.Path(r"C:\Users\alexl\Documents\Intel\OpenVINO\inference_engine_samples_2015\intel64\Release\cpu_extension.dll")]
    # - Alex's NUC
    cpu_exts = [pathlib.Path(r'c:\Intel\computer_vision_sdk\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll')]
    # - POS Machine
    #cpu_exts = [pathlib.Path(r'C:\Users\Public\Documents\OpenVINO\inference_engine_samples_2017\intel64\Release\cpu_extension.dll')]

    # CNN confidence cut-off. Only bboxes above this confidence threshold are processed
    detection_confidence = 0.97 

    image_path = pathlib.Path(r"./Image_Database_package/")
    image_display_time = 2 #3 seconds

    # SKU needs to be present for at least 70% in 
    # smoothing window to be consider detected
    freq_threshold = 0.7 
                        
    smoothing_windown_size = 1 # Smooth for 1 frames
    image_shape = (1920,1080) #(1024,576) # Dimension of images

    white_list = [3, 8, 10, 12, 18, 19, 20]
    app = App(model_path, cpu_exts, model_name, 'CPU', detection_confidence, image_path, image_display_time, smoothing_windown_size, freq_threshold, image_shape, white_list, logging_level=logging.DEBUG)

    try:
        video_capture = USBCamera(height=544, width=960)
        #video_capture = USBCamera(height=720, width=1280)
        video_capture.start(auto_focus=False)
        auto, exp, con, bri = video_capture.get_properties()
        print('AutoExposure:{}, Exposure:{}, Contrast:{}, Brightness:{}'.format(auto,exp,con,bri))
        app.run_from_video(video_capture)
    except KeyboardInterrupt:
        pass
    video_capture.stop()


