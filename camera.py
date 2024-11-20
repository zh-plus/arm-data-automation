import datetime
import os
import time
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image


class CameraController:
    def __init__(self, host, port=8000):
        """Initialize camera controller with host and port."""
        self.base_url = f"http://{host}:{port}"
        self.stream_url = f"{self.base_url}/stream"
        self.capture_url = f"{self.base_url}/capture"
        self.settings_url = f"{self.base_url}/settings"

        self.stream = None
        self.last_frame = None
        self.save_dir = "captured_images"
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.last_reconnect_time = 0
        self.reconnect_cooldown = 15  # seconds
        self.buffer_size = 5

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

    def clear_buffer(self):
        """Clear the video capture buffer by reading and discarding frames."""
        try:
            if self.stream is None or not self.stream.isOpened():
                return False

            # Read and discard frames until buffer is empty
            for _ in range(self.buffer_size):
                self.stream.grab()

            return True
        except Exception as e:
            print(f"Error clearing buffer: {e}")
            return False

    def is_streaming(self):
        return self.stream is not None

    def start_stream(self):
        """Initialize video stream from camera with retry logic."""
        try:
            if self.stream is not None:
                self.stop_stream()

            # Check if we're in cooldown period
            current_time = time.time()
            if current_time - self.last_reconnect_time < self.reconnect_cooldown:
                print("In reconnection cooldown period")
                return False

            self.stream = cv2.VideoCapture(self.stream_url)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            if not self.stream.isOpened():
                print("Failed to open stream")
                return False

            # Reset reconnection counter on successful connection
            self.reconnect_attempts = 0
            return True

        except Exception as e:
            print(f"Error starting stream: {e}")
            return False

    def stop_stream(self):
        """Stop and release video stream."""
        try:
            if self.stream is not None:
                self.stream.release()
                self.stream = None
                self.last_frame = None
            return True
        except Exception as e:
            print(f"Error stopping stream: {e}")
            return False

    def get_frame(self):
        """Get current frame from camera stream with error handling."""
        try:
            if self.stream is None or not self.stream.isOpened():
                return self.last_frame  # Return last good frame while reconnecting

            ret, frame = self.stream.read()
            if not ret:
                print("Failed to read frame")
                self.stop_stream()
                self.start_stream()
                return self.last_frame

            # Check if frame is valid
            if frame is None or frame.size == 0:
                print("Invalid frame received")
                return self.last_frame

            # Convert BGR to RGB for Gradio
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame = frame_rgb
            return frame_rgb

        except cv2.error as e:
            print(f"OpenCV error getting frame: {e}")
            raise e
        except Exception as e:
            print(f"Error getting frame: {e}")
            raise e

    def save_image(self):
        """Save current frame with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.save_image_with_name(f"image_{timestamp}")

    def save_image_with_name(self, filename):
        """
        Save current frame with custom filename.

        Args:
            filename (str): Desired filename (without extension)

        Returns:
            (str, str): Success/failure message, saved path
        """
        try:
            if self.last_frame is None:
                # If no stream frame, try to capture a single image
                try:
                    response = requests.get(self.capture_url, timeout=5)
                    response.raise_for_status()
                except requests.RequestException as e:
                    return f"Failed to capture image: {str(e)}", ''

                # Convert response content to image
                image = Image.open(BytesIO(response.content))
                frame = np.array(image)
            else:
                frame = self.last_frame

            # Ensure filename has .jpg extension
            if not filename.lower().endswith(('.jpg', '.jpeg')):
                filename = f"{filename}.jpg"

            # Create full path
            filepath = os.path.join(self.save_dir, filename)

            # Save image
            if isinstance(frame, np.ndarray):
                # For numpy array (from video stream)
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                # For PIL Image (from direct capture)
                image.save(filepath)

            return f"Image saved successfully as {filename}", filepath

        except Exception as e:
            return f"Error saving image: {str(e)}", ''

    def set_camera_property(self, property_name, value):
        """Set camera property via API with error handling."""
        try:
            response = requests.post(
                f"{self.settings_url}/{property_name}",
                json={"value": value},
                timeout=5
            )
            response.raise_for_status()
            return f"{property_name.capitalize()} set to {value}"
        except requests.RequestException as e:
            return f"Failed to set {property_name}: {str(e)}"
        except Exception as e:
            return f"Error setting {property_name}: {str(e)}"

    def get_camera_property(self, property_name):
        """Get current camera property value with error handling."""
        try:
            response = requests.get(
                f"{self.settings_url}/{property_name}",
                timeout=5
            )
            response.raise_for_status()
            return response.json().get("value")
        except requests.RequestException as e:
            print(f"Failed to get {property_name}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error getting {property_name}: {str(e)}")
            return None
