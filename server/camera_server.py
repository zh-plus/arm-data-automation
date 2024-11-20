import asyncio
import os
from datetime import datetime

import cv2
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


class PropertyValue(BaseModel):
    value: float


class CameraServer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.save_dir = 'captured_images'

        # Initialize camera with decent defaults
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Map property names to OpenCV property IDs
        self.property_map = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
        }

    def generate_frames(self):
        """Generator function for streaming frames."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame to JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def capture_frame(self):
        """Capture a single frame and return as bytes."""
        ret, frame = self.cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        return None

    def save_image(self):
        """Save current frame and return status."""
        ret, frame = self.cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, frame)
            return {"success": True, "message": f"Image saved as {filename}"}
        return {"success": False, "message": "Failed to capture image"}

    def set_camera_property(self, property_name: str, value: float):
        """Set a camera property by name."""
        if property_name not in self.property_map:
            return {"success": False, "message": f"Unknown property: {property_name}"}

        prop_id = self.property_map[property_name]
        success = self.cap.set(prop_id, value)
        return {
            "success": success,
            "message": f"{property_name} {'set successfully' if success else 'failed to set'}"
        }

    def get_camera_property(self, property_name: str):
        """Get a camera property value by name."""
        if property_name not in self.property_map:
            return {"success": False, "message": f"Unknown property: {property_name}"}

        prop_id = self.property_map[property_name]
        value = self.cap.get(prop_id)
        return {"success": True, "value": value}

    def __del__(self):
        """Release camera resources on deletion."""
        self.cap.release()


# Create FastAPI app
app = FastAPI()
camera = CameraServer()


@app.get("/stream")
async def video_stream():
    """Endpoint for MJPEG streaming with error handling and frame rate control."""
    return StreamingResponse(
        camera.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/capture")
async def capture_image():
    """Endpoint for capturing a single image."""
    frame_data = camera.capture_frame()
    if frame_data is not None:
        return Response(content=frame_data, media_type="image/jpeg")
    return JSONResponse(
        content={"error": "Failed to capture image"},
        status_code=500
    )


@app.post("/settings/{property_name}")
async def set_property(property_name: str, value: PropertyValue):
    """Endpoint for setting camera properties."""
    result = camera.set_camera_property(property_name, value.value)
    if not result["success"]:
        return JSONResponse(content=result, status_code=400)
    return JSONResponse(content=result)


@app.get("/settings/{property_name}")
async def get_property(property_name: str):
    """Endpoint for getting camera properties."""
    result = camera.get_camera_property(property_name)
    if not result["success"]:
        return JSONResponse(content=result, status_code=400)
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
