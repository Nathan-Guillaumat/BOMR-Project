import numpy as np
import cv2
import matplotlib.pyplot as plt

def locate_thymio(frame):
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color in HSV
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

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
        dx = front_center[0] - back_center[0]
        dy = front_center[1] - back_center[1]
        orientation = np.arctan2(dy, dx)

        return back_center, orientation
    else:
        return None


def main():
    # Open the video capture
    cap = cv2.VideoCapture(1)

    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Apply the locate_thymio function
        result = locate_thymio(frame)

        # If the function returned a valid position
        if result is not None:
            back_center, orientation = result

            # Draw a circle at the position
            cv2.circle(frame, (int(back_center[0]), int(back_center[1])), 5, (0, 255, 0), -1)

            # Draw a line indicating the orientation
            # Note: You'll need to define how to calculate end_x and end_y based on the orientation
            end_x = int(back_center[0]) + 100 * np.cos(orientation)
            end_y = int(back_center[1]) + 100 * np.sin(orientation)
            cv2.line(frame, (int(back_center[0]), int(back_center[1])), (int(end_x), int(end_y)), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
