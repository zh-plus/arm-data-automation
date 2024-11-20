import base64
import json
import os
import time

from openai import OpenAI

from camera import CameraController
from prompts import detection_prompt, judge_prompt
from robot import RobotController


class RobotVLM:
    camera: CameraController

    def __init__(
            self,
            robot_controller: RobotController,
            api_key: str = None,
            base_url: str = '',
    ):
        """
        Initialize the Robot Vision Agent.

        Args:
            robot_controller: Controller for robot operations
            api_key: API key for vision model access
            base_url: Base URL for the vision model API
            font_path: Path to font file for visualization
        """
        self.robot = robot_controller
        self.camera = self.robot.camera
        self.client = OpenAI(api_key=api_key if api_key else os.environ['OPENAI_API_KEY'],
                             base_url=base_url if base_url else None)

        self.max_retries = 4

    def detect_names(self, command: str, image_path: str) -> dict:
        """
        Call the vision model API with retries.

        Args:
            command: User instruction
            image_path: Path to the image file
            max_retries: Maximum number of retry attempts

        Returns:
            dict: Parsed response from the vision model
        """
        # Encode image as base64
        with open(image_path, 'rb') as image_file:
            image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": detection_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": command},
                    {"type": "image_url", "image_url": {"url": image}}
                ]
            }
        ]

        for attempt in range(self.max_retries):
            try:
                print(f'    Attempt {attempt + 1} of {self.max_retries} to access vision model')
                completion = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    response_format={"type": "json_object"}
                )

                result = json.loads(completion.choices[0].message.content.strip())
                print('    Vision model call successful!')
                return result

            except Exception as e:
                print(f'    Vision model error on attempt {attempt + 1}: {e}')
                if attempt == self.max_retries - 1:
                    raise RuntimeError("Failed to get valid response from vision model")
                time.sleep(1)  # Brief pause before retry

    def judge(self, user_prompt, image_path):
        with open(image_path, 'rb') as image_file:
            image = 'data:image/jpeg;base64,' + base64.b64encode(image_file.read()).decode('utf-8')

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": judge_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image}}
                ]
            }
        ]

        for attempt in range(self.max_retries):
            print(f'    Attempt {attempt + 1} of {self.max_retries} to access vision model')
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            result = completion.choices[0].message.content.strip()
            print('    Vision model call successful!')

            if not ('yes' in result.lower() or 'no' in result.lower()):
                print(f'Attempt {attempt + 1} fail:  {result}')
                continue

            return result
