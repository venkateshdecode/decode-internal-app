import cv2
import os
import glob 
import numpy as np 
from typing import List
from PIL import Image

class ImageFlow():
    def set_hsv_template(self, image_template: np.ndarray):

        self.hsv_template = image_template
                
    def calc_flow(self, im1: np.ndarray, im2: np.ndarray): 
        return cv2.calcOpticalFlowFarneback(im1,im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def get_motion(self, im1: np.ndarray, im2: np.ndarray):
        flow = self.calc_flow(im1, im2)
        mag, _ = self._cart_to_pol(flow[...,0], flow[...,1]) #fix cv2 bug
        return np.mean(mag)

    def visualize_flow(self, im1: np.ndarray, im2: np.ndarray, return_motion=False):
        
        flow = self.calc_flow(im1, im2)    
        mag, ang = self._cart_to_pol(flow[...,0], flow[...,1])
        hsv = np.zeros_like(self.hsv_template)
        hsv[...,1] = 255                 
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        if return_motion: 
            return cv2.resize(gray, (im1.shape[1], im1.shape[0] ), interpolation = cv2.INTER_CUBIC ), np.mean(mag)
        else:
            return cv2.resize(gray, (im1.shape[1], im1.shape[0] ), interpolation = cv2.INTER_CUBIC )

    @staticmethod
    def _cart_to_pol(x, y):
        ang = np.arctan2(y, x)
        mag = np.hypot(x, y)
        return mag, ang
