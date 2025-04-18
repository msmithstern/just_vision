import cv2 # opencv-based functions
import time
import math
import numpy as np
from scipy import ndimage
from skimage import io
from skimage import img_as_float32, img_as_ubyte
from skimage.color import rgb2gray

'''
Creates an instance of the webcam by running 'python input.py'. Referenced from hw2 and cv2 documentation. 
'''
class webcam():
    window_name = "Final Project"
    def __init__(self, **kwargs):
        self.writeWelcome()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # Initialize camera
        # The argument is the device id. 
        # If you have more than one camera, you can access them 
        # by passing a different id, e.g., cv2.VideoCapture(1)

        #self.video_camera = cv2.VideoCapture(0)
        self.cam = cv2.VideoCapture(0)

        # Get the default frame width and height
        frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

        while True:
            ret, frame = self.cam.read()

            # Write the frame to the output file
            out.write(frame)

            # Display the captured frame
            cv2.imshow(self.window_name, frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture and writer objects
        self.cam.release()
        out.release()
        cv2.destroyAllWindows()

    def writeWelcome(self):
        print(""" 
            Welcome to our final project! 
        """)
        return

if __name__ == '__main__':
    webcam()