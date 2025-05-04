import cv2
import numpy as np
from pylibfreenect2 import Freenect2, SyncMultiFrameListener, FrameType
import time
import datetime
import threading
from pydub import AudioSegment
import simpleaudio as sa

# Replace this with your actual image processing function
def process_captured_image(image, timestamp_ms):
    print(f"Processing image captured at {timestamp_ms} ms")

    print(""" 
            Capturing data
          
          ________________________________
        """)
    # Placeholder for any processing logic
    # Example: analyze motion, save analysis, etc.
    # You could save or return results here

class KinectDepthImageCapture:
    def __init__(self, capture_timestamps_ms):
        self.window_name = "Kinect Depth Capture"
        self.frame_width = 512
        self.frame_height = 424
        self.capture_timestamps_ms = sorted(capture_timestamps_ms)
        self.audio_file = 'audio_only.mp3'
        self.audio_start_time = None
        self.next_capture_index = 0
        self.player = None
        self.setup_kinect()

    def setup_kinect(self):
        self.fn = Freenect2()
        if self.fn.enumerateDevices() == 0:
            raise RuntimeError("No Kinect device detected!")

        serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(serial)
        self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.startStreams(rgb=True, depth=True)

    def play_audio(self):
        audio = AudioSegment.from_file(self.audio_file)
        self.player = sa.play_buffer(audio.raw_data, audio.channels, audio.sample_width, audio.frame_rate)
        self.audio_start_time = time.time()  # Note the start time

    def run(self):
        print("Starting audio playback and capture loop...")
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Start audio playback in a separate thread
        audio_thread = threading.Thread(target=self.play_audio)
        audio_thread.start()

        try:
            while self.player is None or not self.player.is_playing():
                time.sleep(0.01)  # Wait for player to start

            #the second condition causes the program to stop after there's no more time stamps 

            while self.player.is_playing() and self.next_capture_index < len(self.capture_timestamps_ms): 
                frames = self.listener.waitForNewFrame()
                depth_frame = frames["depth"]
                depth_image = np.asarray(depth_frame.asarray())

                depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow(self.window_name, depth_normalized)
                key = cv2.waitKey(1)

                elapsed_ms = int((time.time() - self.audio_start_time) * 1000)

                # Check if it's time to capture the next image
                if elapsed_ms >= self.capture_timestamps_ms[self.next_capture_index]:
                    print(f"Capturing image at {elapsed_ms} ms")
                    self.save_and_process(depth_normalized, elapsed_ms)
                    self.next_capture_index += 1

                if key == ord('q') or key == 27:
                    break

                self.listener.release(frames)

        finally:
            self.cleanup()

    def save_and_process(self, depth_normalized, timestamp_ms):
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f'capture_{timestamp_str}_{timestamp_ms}ms.png'
        cv2.imwrite(filename, depth_normalized)
        print(f"Saved depth image to {filename}")
        process_captured_image(depth_normalized, timestamp_ms)

    def cleanup(self):
        print("Cleaning up...")
        self.device.stop()
        self.device.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example: Capture at 5s, 10s, 15s (in ms)
    capture_times = [5000, 10000, 15000]
    KinectDepthImageCapture(capture_timestamps_ms=capture_times).run()