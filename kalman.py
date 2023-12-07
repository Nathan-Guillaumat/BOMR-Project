import os
import sys
import math
from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from vision_final import *

import cv2
import time

from threading import Timer

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm


def kalman_filter( x_est_prev, P_est_prev, Vr, Vl, Ts, camera_x=0, camera_y=0, camera_angle=0,):

    ## Prediciton through the a priori estimate
    # estimated mean of the state
    if abs(Vr-Vl) < 30:
        speed_conv_factor = speed_conv_factor_straight
    elif Vr<Vl:
        speed_conv_factor = speed_conv_factor_right
    else:
        speed_conv_factor = speed_conv_factor_left
    
    orientation = np.double(x_est_prev[2][0]%360)*2*np.pi/360
    #x_est_prev[2][0] = orientation
    B = np.array([[Ts*np.cos(orientation)/2,Ts*np.cos(orientation)/2],
         [Ts*np.sin(orientation)/2,Ts*np.sin(orientation)/2],
         [-Ts*360/(l*2*np.pi), Ts*360/(l*2*np.pi)]])
    U = np.array([[Vr*speed_conv_factor],[Vl*speed_conv_factor]])
    x_est_a_priori = np.dot(A, x_est_prev) + np.dot(B, U);

    # Estimated covariance of the state
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T));
    P_est_a_priori = P_est_a_priori + Q if type(Q) != type(None) else P_est_a_priori
        
    ## Update         
    # y, C, and R for a posteriori estimate, depending on transition
    if (camera_x != 0) : #XOR (one or the other but not both)
        # transition detected aka camera provides the position
        y = np.array([[camera_x],[camera_y],[camera_angle]])
        H = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
        R = np.array([[rpx, 0, 0],[0, rpy,0],[0, 0, r_theta_nu]]) #not the value of the variance of the position measurments
        
        # innovation / measurement residual
        i = y - np.dot(H, x_est_a_priori); #measured-mean aka difference between the two states
        # measurement prediction covariance
        S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R;
                 
        # Kalman gain (tells how much the predictions should be corrected based on the measurements)
        K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S)));
            
        # a posteriori estimate
        x_est = x_est_a_priori + np.dot(K,i);
        P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori));

    else:
        x_est = x_est_a_priori
        P_est = P_est_a_priori
     
    return x_est, P_est

def filter():
    global x_est, P_est, Ts, angle, Vr, Vl, frame, image_ready

    thymio_data.append({"left_speed":node["motor.left.speed"],
                        "right_speed":node["motor.right.speed"]}) 

    Vl = thymio_data[-1]["left_speed"]
    Vr = thymio_data[-1]["right_speed"]

    camera_x = 0
    camera_y = 0
    camera_angle = 0

    if image_ready:
        thymio_result = locate_thymio(frame)
        camera_x = thymio_result[0][0][0]*0.4
        camera_y = thymio_result[0][0][1]*0.4
        camera_angle = (thymio_result[2])*360/(2*np.pi)
        image_ready = False

    x_est, P_est = kalman_filter(x_est, P_est, Vr, Vl, Ts, camera_x, camera_y, camera_angle)
    camera_x = 0 
