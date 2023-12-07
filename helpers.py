import os
import sys
import math
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from vision import *

import cv2
import time

from threading import Timer

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
import asyncio

def angle_to_rotate(pos_i, pos_f, theta_i):
    '''
    angle to turn between two points with an initial orientation
    
    args: initial_position, final_position, initial_orientation
    return: angle to rotate in degree
    '''
    a = pos_f-pos_i
    theta_f = np.arctan2(a[1], a[0])
    r_theta = (theta_f*180)/np.pi-theta_i
    if r_theta > 180:
        r_theta -= 360
    elif r_theta < -180:
        r_theta += 360
    return r_theta

def distance_to_cover(pos_i, pos_f):
    '''
    distance between two points

    args: initial position
    out: final position
    '''
    distance = np.linalg.norm(pos_f-pos_i)
    return distance

class RepeatedTimer(object):
    '''
    class to set a repeated thred timer

    args: object to repeat
    '''
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

def motors(left, right):
    '''
    function to set the motors

    args: motor speed for the right, motor speed for the left 
    out: motors speeds to set the Thymio
    '''
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }
