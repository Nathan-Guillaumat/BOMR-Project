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
import asyncio

def test_saw_obstacle(threshold, verbose=True):
    print("\t Test saw obstacle: ")
    if any([x > threshold for x in node['prox.horizontal'][0:5]]):
        if verbose: print("\t\t Obstacle detected")
        return True
    
    return False

async def local_avoidance(motor_speed=100, threshold=0, verbose=False):
    obstacleSpeedGain = [6, 4, -2, -6, -8]
    saw_obstacle = True
    speedL = motor_speed
    speedR = motor_speed
    prev_state = "global"

    if verbose:
        print("Local avoidance")

    while saw_obstacle:
        if test_saw_obstacle(threshold, verbose=False):
            if prev_state == "global":
                prev_state = "local"
                print("\t Transitioning to local avoidance")

            for i in range(5):
                speedL += node["prox.horizontal"][i] * obstacleSpeedGain[i] // 100
                speedR += node["prox.horizontal"][i] * obstacleSpeedGain[4 - i] // 100

          
            await node.set_variables(motors(speedL, speedR))
            if verbose:
                print("\t\t Adjusting for obstacle")

        else:
            saw_obstacle = False
            if prev_state == "local":
                #prev_state = "global"
                print(prev_state)
                print("\t Transitioning to global navigation")
                
            

            # Reset motor speeds when no obstacles are detected
            speedL = motor_speed
            speedR = motor_speed
            await node.set_variables(motors(speedL, speedR))
            
            # Add a small delay to avoid continuously checking for obstacles too quickly
            await client.sleep(0.1)


async def path_following(motor_speed=100, threshold=0,  verbose=False):
    
    global n, positions, x_est, P_est, Ts, read, Vr, Vl, n

    saw_obstacle = False
    
    if verbose: print("\t Path following ")
    await node.set_variables(motors(motor_speed, motor_speed))
           
    prev_state="local"
    
    while not saw_obstacle:
        
        if test_saw_obstacle(threshold, verbose=False):
            if prev_state=="local": 
                prev_state="global"
            saw_obstacle = True
            await client.sleep(0.1) #otherwise, variables would not be updated
        else:
            start = time.time()
            if (positions[n][0]+position_threshold > x_est[0][0]) and (positions[n][0]-position_threshold < x_est[0][0]):
                if (positions[n][1]+position_threshold > x_est[1][0]) and (positions[n][1]-position_threshold < x_est[1][0]):
                    n+=1
            angle = angle_to_rotate([x_est[0][0],x_est[1][0]],positions[n],x_est[2][0]);
            if (abs(angle)<angle_threshold):
                #Vr = 100
                #Vl = 100
                drive_task = asyncio.create_task(drive_for_seconds(client, Ts))
                await drive_task
            else:
                if angle>0:
                    right = 1
                    #Vr = -100
                    #Vl = 100
                else:
                    right = 0
                    #Vr = 100
                    #Vl = -100

                turn_task = asyncio.create_task(turn_for_seconds(client, Ts, right))
                await turn_task
            #print(x_est)
            stop = time.time()
            #print("time elapsed : ", start - stop)
            
    return 