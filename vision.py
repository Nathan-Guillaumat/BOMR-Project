import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def locate_goal(frame):
    '''   
    locate goal
    
    args: frame
    return: center coordinates of goal
    '''
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds for the green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    # Threshold the image to get only red colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check if any contours are found
    if contours:
        # Find the contour with the maximum area (largest red object)
        max_contour = max(contours, key=cv2.contourArea)
        # Calculate the center of the bounding box of the contour
        x, y, w, h = cv2.boundingRect(max_contour)
        center_x = x + w // 2
        center_y = y + h // 2
        return center_x, center_y
    else:
        return None

def locate_thymio(frame):
    '''   
    locate thymio robot
    
    args: frame
    return: center coordinates of back-dot, of front-dot, orientation in rad
    '''
    
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the two masks
    mask = cv2.bitwise_or(mask1, mask2)
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        # Sort contours by area
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # The largest contour is the back dot and the second largest is the front dot
        back_dot = sorted_contours[0]
        front_dot = sorted_contours[1]

        # Calculate the centers of the two dots
        back_center = np.mean(back_dot, axis=0)
        front_center = np.mean(front_dot, axis=0)

        # The orientation of the robot is the angle of the line connecting the two centers relative to the x-axis
        dx = front_center[0][0] - back_center[0][0]
        dy = front_center[0][1] - back_center[0][1]  
        orientation = np.arctan2(dy, dx)

        return back_center, front_center, orientation
    else:
        return [0,0], 0

def filter_contours(contours, n_clusters=2):
    '''   
    K-means to help locate_static_obstacles find the right objects
    
    args: frame, n_clusters=2 (constant)
    return: list of the largest contours (by area) between the n=2 clusters
    '''
    # Calculate the areas of the contours
    areas = np.array([cv2.contourArea(contour) for contour in contours]).reshape(-1, 1)
    # Apply K-means clustering to the areas
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(areas)
    # Find the cluster with the largest mean area
    largest_cluster = np.argmax([np.mean(areas[kmeans.labels_ == i]) for i in range(n_clusters)])
    # Filter out contours that are not in the largest cluster
    large_contours = [contour for contour, label in zip(contours, kmeans.labels_) if label == largest_cluster]
    return large_contours

def locate_static_obstacles(frame, d):
    '''   
    locate static obstacles
    
    args: frame, distance to enlarge obstacles
    return: nested list of obstacle corners (given in clockwise order)
    '''
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds for the black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    # Threshold the image to get only black colors
    mask = cv2.inRange(hsv, lower_black, upper_black)
    # Threshold the image to get only blue colors
    mask = cv2.inRange(hsv, lower_black, upper_black)
    # Initialize lists to store the centers, corners, and contours of the obstacles
    obstacle_centers = []
    obstacle_corners = []
    obstacle_contours = []
    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours(contours) # k-means
    # check if any contours are found
    if contours:
        # Iterate over the contours
        for contour in contours:
            # Calculate the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Enlarge the bounding rectangle by d
            x -= d
            y -= d
            w += 2*d
            h += 2*d
            # Calculate the center of the bounding rectangle
            center_x = x + w // 2
            center_y = y + h // 2
            # Add the center, corners, and contour to the respective lists
            obstacle_centers.append((center_x, center_y))
            obstacle_corners.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]) # always clockwise
            obstacle_contours.append(contour)

    return obstacle_corners
