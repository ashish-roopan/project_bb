import mss
import cv2
import numpy as np
import time
# import threading

class ScreenCapture:
    def __init__(self, debug=False):
        self.w, self.h = 1920, 1080
        self.debug = debug
        self.t0 = time.time()
        self.n_frames = 1
        self.monitor = {"top": 0, "left": 1080, "width": self.w, "height": self.h} #Only one monitor
        self.sct = mss.mss()
        self.frame = None

    def read(self):
        img = self.sct.grab(self.monitor)
        img = np.array(img)                        
        self.frame = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        return self.frame
    
    # def start(self):
    #     self.screen_capture_thread = threading.Thread(target=self.screen_capture)
    #     self.screen_capture_thread.start()
    #     print("Screen Capture started")

