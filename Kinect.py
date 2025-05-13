import cv2
import numpy as np
from pylibfreenect2 import Freenect2, SyncMultiFrameListener, FrameType
import time
import datetime
import threading
from pydub import AudioSegment
import simpleaudio as sa
import os
import random
import isolation_mask as mask
from pose_classifier import just_dance_score

def print_update(timestamp_ms):
    """
        Prints a message to the terminal that image was captured

        Parameters: 
        time stamp for when the image was taken
    """
    dance_icons = [
        """           
            \o/        o/      \o        o/
             |        <|        |\\      /|
            / \\       / >      / \\      / \\      """,
            """           
            \o/        o/       o/        \o     
                |        <|       /|          |\\   
            / \\       / >      / \\        / \\         """,
            """           
            \o/       o/        \o         o/      
                |       /|          |\\       <|      
            / \\      / \\        / \\       / >          """,
            """           
                o/      \o/        o/         \o
            /|        |        <|           |\\
            / \\      / \\       / >         / \\         """
        ]
    print()
    print(f""" 
            Processing image captured at {timestamp_ms} ms
          ________________________________

           {dance_icons[random.randint(0,0)]}
           
           ________________________________
        """)
    print()


class KinectDepthImageCapture:
    """
        This class runs and captures depth images from the Kinect camera. 

        Parameters: 
        capture_timestamps_ms: a list of time stamps to capture images
        video_file: the video to play as a dance tutorial
        audio_file: the audio to play for the dance
    """
    def __init__(self, capture_timestamps_ms, video_file='demo_video.mp4', audio_file='demo_audio.mp3'):
        self.window_name = "Kinect Depth Capture"
        self.video_window_name = "Dance Video"
        self.frame_width = 512
        self.frame_height = 424
        self.capture_timestamps_ms = sorted(capture_timestamps_ms)
        self.video_file = video_file
        self.audio_file = audio_file
        self.audio_start_time = None
        self.next_capture_index = 0
        self.player = None
        self.video_cap = None
        self.setup_kinect()
        self.setup_video()

    def setup_kinect(self):
        """
            This method sets up the Kinect camera, ensuring a device is connected and can capture input
        """
        self.fn = Freenect2()
        if self.fn.enumerateDevices() == 0:
            raise RuntimeError("No Kinect device detected!")

        serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(serial)
        self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.startStreams(rgb=True, depth=True)

    
    def setup_video(self):
        """
            This method sets up the video to play, ensuring the file exists and can be opened in the appropriate frame speed
        """
        # Check if video file exists
        if not os.path.exists(self.video_file):
            print(f"Warning: Video file '{self.video_file}' not found. Video playback will be disabled.")
            return

        self.video_cap = cv2.VideoCapture(self.video_file)
        if not self.video_cap.isOpened():
            print(f"Error: Could not open video file '{self.video_file}'")
            self.video_cap = None
            return

        # Get video properties
        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.video_frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = self.video_frame_count / self.video_fps * 1000  # duration in ms
        
        print(f"Video loaded: {self.video_fps} FPS, {self.video_frame_count} frames, {self.video_duration/1000:.2f} seconds")

    def play_audio(self):
        """
            This method plays the audio needed for the dance to run
        """
        audio = AudioSegment.from_file(self.audio_file)
        self.player = sa.play_buffer(audio.raw_data, audio.channels, audio.sample_width, audio.frame_rate)
        self.audio_start_time = time.time()

    def run(self):
        """
            This method runs a loop that starts and displays the camera, audio, and video 
            information. Once the audio and video are running, we display the isolated human
            body input for the user to see. At appropriate time stamps, we capture the depth image
            to use in joint estimation and classification later on. 
        """
        print("Starting audio/video playback and capture loop...")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Create video window if video is available
        if self.video_cap is not None:
            cv2.namedWindow(self.video_window_name, cv2.WINDOW_NORMAL)

        # Start audio in a separate thread
        audio_thread = threading.Thread(target=self.play_audio)
        audio_thread.start()

        try:
            # Wait for audio to start
            while self.player is None or self.audio_start_time is None:
                time.sleep(0.01)

            while (self.player.is_playing() and 
                   self.next_capture_index < len(self.capture_timestamps_ms)):
                
                # Get depth frame from Kinect
                frames = self.listener.waitForNewFrame()
                depth_frame = frames["depth"]
                depth_image = np.asarray(depth_frame.asarray())
                _, _, isolated = mask.isolate_person(depth_image)

                # Normalize and display the isolated (masked) depth image
                depth_normalized = cv2.normalize(isolated, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow(self.window_name, depth_normalized)

                
                # Calculate current time position in milliseconds
                elapsed_ms = int((time.time() - self.audio_start_time) * 1000)
                
                # Display video frame if available
                if self.video_cap is not None:
                    # Calculate the frame position based on elapsed time
                    frame_pos = int(elapsed_ms / 1000.0 * self.video_fps)
                    
                    # Set video position and get frame
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, video_frame = self.video_cap.read()
                    
                    if ret:
                        cv2.imshow(self.video_window_name, video_frame)
                
                # Check if it's time to capture an image
                if (self.next_capture_index < len(self.capture_timestamps_ms) and 
                    elapsed_ms >= self.capture_timestamps_ms[self.next_capture_index]):
                    print(f"Capturing image at {elapsed_ms} ms")
                    self.save_and_process(isolated, elapsed_ms)
                    self.next_capture_index += 1

                # Check for user input to exit
                key = cv2.waitKey(1)
                if key == ord('q'): 
                    break

                self.listener.release(frames)

        finally:
            self.cleanup()

    def save_and_process(self, depth_image, timestamp_ms):
        """
            This method saves the depth images to a folder called "dancing". 

            Parameters: 
            depth_image: image to save 
            timestamp_ms: timestamp used to create unique name for each image
        """
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir = 'dancing/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        filename = dir + f'masked_capture_{timestamp_str}_{timestamp_ms}ms.npy'
        np.save(filename, depth_image.astype(np.float32))
        print(f"Saved depth image to {filename}")
        print_update(timestamp_ms)

    def cleanup(self):
        """
            This method cleans up the camera after exiting. 
        """
        print("Cleaning up...")
        if self.player and self.player.is_playing():
            self.player.stop()
        if self.video_cap is not None:
            self.video_cap.release()
        self.device.stop()
        self.device.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    """
        This is the main method cal that establishes timestamps to 
        capture poses, calls an instance of the Kinect camera, and 
        calls the method to score the player's dance. 
    """
    capture_times = [
        3057, 6007, 7007, 12000, 12044, 13047, 14042, 15043,
        19038, 20023, 21019, 22015, 24001, 24047, 25034, 26003, 27003, 27052,
        28052, 29049, 30052, 34041, 35032, 36032, 37029, 
        39014, 39056, 40048, 41015, 42046, 46006, 46051, 47014, 47041, 
        49048, 50036, 51004, 51028, 53028, 53031, 54020, 54044, 55010,
        55046, 56006, 56040
    ]

    KinectDepthImageCapture(
        capture_timestamps_ms=capture_times,
        video_file='demo_video.mp4',
        audio_file='demo_audio.mp3', 
    ).run()
    just_dance_score("dancing")