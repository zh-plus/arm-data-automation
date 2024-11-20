from pathlib import Path
from typing import List

import cv2
from PIL import ImageFont
from supervision import Detections

from agent import RobotVLM
from grounded_sam2.grounded_sam2_florence2_open_vocal_detection import open_vocab_detection
from robot import RobotController


class DataAutomation:
    """
    Automated Data Acquisition System for Robotic Arms
    """

    def __init__(self, robot: RobotController):
        self.robot = robot
        self.vlm_agent = RobotVLM(self.robot)
        self.command_pair = [
            ('将绿色方块放到机器人上', '将绿色方块放到黑色三角上')
        ]
        self.font = ImageFont.truetype('asset/SimHei.ttf', 26)

    def start(self):
        print('Start data generation')
        limit = 3
        command_idx = 0
        for i in range(limit):
            print(f'The {i}th data collection. Start the {command_idx}th command')
            success = self.execute_command(self.command_pair[0][command_idx])
            limit -= 1

            if success:
                command_idx = 0 if command_idx else 1

    def execute_command(self, command, check_visualization=False):
        print(f"Executing robot vision instruction: {command}")

        # Step 1: Move robot to zero position
        print("Step 1: Moving robot to zero position")
        self.robot.back_zero()

        # Step 2: Process the instruction
        print(f"Step 2: Processing instruction: {command}")

        # Step 3: Take top view photo
        print("Step 3: Taking top view photo")
        image_path = self.robot.top_view_shot()
        if not image_path or "Error" in image_path:
            raise RuntimeError(f"Failed to capture image: {image_path}")

        # Step 4: Get detection result

        # Step 4.1 Get object names
        print("Step 4: Calling vision model")
        result = self.vlm_agent.detect_names(f'Instruction:\n{command}', image_path)
        names = [result['start'], result['end']]

        # Step 4.2 Get object positions
        print("Step 5: Detecting positions")
        detection_results, detection_viz_paths = open_vocab_detection(image_path, names)
        start_x, start_y, end_x, end_y = self.get_detected_position(detection_results, names)

        # Show detection_viz_paths['masks'] to wait confirm
        self.viz_arrow(detection_viz_paths['masks'], start_x, start_y, end_x, end_y, user_check=True)

        # Step 6: Convert pixel coordinates to robot coordinates
        print("Step 6: Converting coordinates")
        start_x_robot, start_y_robot = self.robot.eye2hand(start_x, start_y)
        end_x_robot, end_y_robot = self.robot.eye2hand(end_x, end_y)

        # Step 7: Execute movement
        print("Step 7: Executing robot movement")
        self.robot.gripper_move(
            XY_START=[start_x_robot, start_y_robot],
            XY_END=[end_x_robot, end_y_robot]
        )

        # Step 8: Judge completion
        print("Step 8: Check if the robot complete task")
        image_path = self.robot.top_view_shot()
        if not image_path or "Error" in image_path:
            raise RuntimeError(f"Failed to capture image: {image_path}")
        success = self.judge_result(result['description'], command, image_path)
        cv2.destroyAllWindows()

        if success:
            print("Complete task")
        else:
            print("Task incomplete")

        return success

    def judge_result(self, initial_description, command, image_path):
        prompt = f'''- Initial description: "{initial_description}"
- Command: "{command}"
- Image'''

        result = self.vlm_agent.judge(prompt, image_path)

        return 'yes' in result.lower()

    def get_detected_position(self, detection_results: Detections, names: List[str]):
        if len(names) > 2:
            print(f'Name more thant 2: {names}!')

        name2xyxy = {}
        name2areas = {}
        for label_id, xyxy, area in zip(detection_results.class_id, detection_results.xyxy, detection_results.area):
            label_name = names[label_id]

            if not name2xyxy.get(label_name):
                name2xyxy[label_name] = xyxy
                name2areas[label_name] = area
            elif area > name2xyxy[label_name]:
                name2xyxy[label_name] = xyxy
                name2areas[label_name] = area

        start_name = names[0]
        start_xyxy = name2xyxy[start_name]
        start_x = (start_xyxy[0] + start_xyxy[2]) / 2
        start_y = (start_xyxy[1] + start_xyxy[3]) / 2

        end_name = names[1]
        end_xyxy = name2xyxy[end_name]
        end_x = (end_xyxy[0] + end_xyxy[2]) / 2
        end_y = (end_xyxy[1] + end_xyxy[3]) / 2

        return start_x, start_y, end_x, end_y

    def viz_arrow(self, image_path: Path, start_x, start_y, end_x, end_y, user_check=False):
        """Draw an arrow on the image, from start to end. Display it, and if user_check=True, let user to decide if continue (press c to continue, q to quit)"""

        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Make a copy to avoid modifying the original
        display_image = image.copy()

        # Define arrow parameters
        arrow_color = (0, 255, 0)  # Green color in BGR
        thickness = 2
        tip_length = 0.3  # Length of the arrow tip relative to the arrow length

        # Draw the arrow
        cv2.arrowedLine(
            display_image,
            (int(start_x), int(start_y)),
            (int(end_x), int(end_y)),
            color=arrow_color,
            thickness=thickness,
            tipLength=tip_length
        )

        # Display the image
        window_name = 'Image with Arrow'
        cv2.imshow(window_name, display_image)

        if user_check:
            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord('c'):  # Continue
                    cv2.destroyWindow(window_name)
                    return display_image
                elif key == ord('q'):  # Quit
                    cv2.destroyWindow(window_name)
                    raise InterruptedError("User chose to quit")
        else:
            # If no user check required, wait briefly and close
            cv2.waitKey(1000)
            cv2.destroyWindow(window_name)
            return display_image
