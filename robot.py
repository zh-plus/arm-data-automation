import time

import numpy as np
from pymycobot import MyCobot280Socket

from camera import CameraController


class RobotController:
    def __init__(self, mc: MyCobot280Socket, camera_controller: CameraController = None):
        """Initialize robot connection, GPIO, and camera controller"""
        print('Connecting to robot...')
        self.mc = mc
        self.camera = camera_controller
        self.init_robot()
        self.init_gpio()

    def init_gpio(self):
        """Initialize GPIO settings"""
        self.mc.set_gpio_mode("BCM")  # Set BCM mode
        # Setup pins as outputs
        self.mc.set_gpio_out(20, "out")  # Pump solenoid valve
        self.mc.set_gpio_out(21, "out")  # Release valve
        # Initialize pump valve as closed
        self.mc.set_gpio_output(20, 1)

    def init_robot(self):
        """Initialize robot settings"""
        # Set interpolation mode (equivalent to former fresh mode 0)
        self.mc.set_fresh_mode(0)

    def pump_on(self):
        """Turn on pump"""
        print('    Opening pump')
        self.mc.set_gpio_output(20, 0)  # Turn on pump solenoid valve

    def pump_off(self):
        """Turn off pump with air release sequence to ensure object drops"""
        print('    Closing pump')
        self.mc.set_gpio_output(20, 1)  # Close pump solenoid valve
        time.sleep(0.05)
        self.mc.set_gpio_output(21, 0)  # Open release valve
        time.sleep(0.2)
        self.mc.set_gpio_output(21, 1)  # Close release valve
        time.sleep(0.05)
        self.mc.set_gpio_output(21, 0)  # Open release valve again for second release
        time.sleep(0.2)
        self.mc.set_gpio_output(21, 1)  # Close release valve
        time.sleep(0.05)

    def gripper_open(self):
        print('    Opening gripper')
        self.mc.set_gripper_state(0, 80)
        time.sleep(0.6)

    def gripper_close(self):
        print('    Closing gripper')
        self.mc.set_gripper_state(1, 80)
        time.sleep(1)

    def back_zero(self):
        """Return robot to zero position"""
        print('Moving robot to zero position')
        self.mc.send_angles([0, 0, 0, 0, 0, 0], 20)
        time.sleep(2)

    def relax_arms(self):
        """Relax all robot joints"""
        print('Relaxing robot joints')
        self.mc.release_all_servos()

    def head_shake(self):
        """Perform head shake movement"""
        self.mc.send_angles([0.87, -50.44, 47.28, 0.35, -0.43, -0.26], 70)
        time.sleep(1)
        for _ in range(2):
            self.mc.send_angle(5, 30, 80)
            time.sleep(0.5)
            self.mc.send_angle(5, -30, 80)
            time.sleep(0.5)
        self.mc.send_angles([0, 0, 0, 0, 0, 0], 40)
        time.sleep(2)

    def head_dance(self):
        """Perform dance movement sequence"""
        self.mc.send_angles([0.87, -50.44, 47.28, 0.35, -0.43, -0.26], 70)
        time.sleep(1)
        dance_sequences = [
            [(-0.17), (-94.3), 118.91, (-39.9), 59.32, (-0.52)],
            [67.85, (-3.42), (-116.98), 106.52, 23.11, (-0.52)],
            [(-38.14), (-115.04), 116.63, 69.69, 3.25, (-11.6)],
            [2.72, (-26.19), 140.27, (-110.74), (-6.15), (-11.25)],
            [0, 0, 0, 0, 0, 0]
        ]
        for sequence in dance_sequences:
            self.mc.send_angles(sequence, 80)
            time.sleep(1.7)

    def head_nod(self):
        """Perform nodding movement"""
        self.mc.send_angles([0.87, -50.44, 47.28, 0.35, -0.43, -0.26], 70)
        for _ in range(2):
            self.mc.send_angle(4, 13, 70)
            time.sleep(0.5)
            self.mc.send_angle(4, -20, 70)
            time.sleep(1)
            self.mc.send_angle(4, 13, 70)
            time.sleep(0.5)
        self.mc.send_angles([0.87, -50.44, 47.28, 0.35, -0.43, -0.26], 70)

    def move_to_coords(self, X=150, Y=-130, HEIGHT_SAFE=230):
        """Move to specified coordinates"""
        print(f'Moving to coordinates: X {X} Y {Y}')
        self.mc.send_coords([X, Y, HEIGHT_SAFE, 0, 180, 90], 20, 0)
        time.sleep(4)

    def move_to_top_view(self):
        """Move to top viewing position"""
        print('Moving to top view position')
        self.mc.send_angles([-79.1, -3.33, -84.28, 0, 1.49, -33.48], 30)

        time.sleep(3)

    def top_view_shot(self):
        """
        Take a photo from top view position using CameraController
        
        Args:
            check (bool): If True, wait for user confirmation before continuing
        
        Returns:
            str: Path to saved image or error message
        """
        if self.camera is None:
            return "Error: Camera controller not initialized"

        print('    Moving to top view position')
        self.move_to_top_view()
        time.sleep(3)  # Give camera time to stabilize

        # Ensure the stream is started
        if not self.camera.stream:
            if not self.camera.start_stream():
                return "Error: Failed to start camera stream"

        # Get frame from camera
        self.camera.clear_buffer()
        frame = self.camera.get_frame()
        if frame is None:
            return "Error: Failed to capture frame"

        # Save image with timestamp
        image_path = self.camera.save_image()[1]
        print('image saved')

        if not image_path:
            return "Error: Failed to save image"

        return image_path

    def eye2hand(self, X_im=160, Y_im=120):
        """Convert image coordinates to robot coordinates"""
        cali_1_im = [64, 434]                       # 左下角，第一个标定点的像素坐标，要手动填！
        cali_1_mc = [122.4, -98.0]                  # 左下角，第一个标定点的机械臂坐标，要手动填！
        cali_2_im = [442, 20]                         # 右上角，第二个标定点的像素坐标
        cali_2_mc = [-33.7, -271.3]                   # 右上角，第二个标定点的机械臂坐标，要手动填！

        X_cali_im = [cali_1_im[0], cali_2_im[0]]
        X_cali_mc = [cali_1_mc[0], cali_2_mc[0]]
        Y_cali_im = [cali_2_im[1], cali_1_im[1]]
        Y_cali_mc = [cali_2_mc[1], cali_1_mc[1]]

        X_mc = int(np.interp(X_im, X_cali_im, X_cali_mc))
        Y_mc = int(np.interp(Y_im, Y_cali_im, Y_cali_mc))

        return X_mc, Y_mc

    def pump_move(self, XY_START=[230, -50], HEIGHT_START=90, XY_END=[100, 220],
                  HEIGHT_END=100, HEIGHT_SAFE=220):
        """Move object using pump from start to end position"""
        print('Starting pump move sequence')

        # Set interpolation mode
        self.mc.set_fresh_mode(0)

        print('    Moving pump above object')
        self.mc.send_coords([XY_START[0], XY_START[1], HEIGHT_SAFE, 0, 180, 90], 20, 0)
        time.sleep(4)

        # Turn on pump
        self.pump_on()

        print('    Lowering pump to grab object')
        self.mc.send_coords([XY_START[0], XY_START[1], HEIGHT_START, 0, 180, 90], 15, 0)
        time.sleep(4)

        print('    Lifting object')
        self.mc.send_coords([XY_START[0], XY_START[1], HEIGHT_SAFE, 0, 180, 90], 15, 0)
        time.sleep(4)

        print('    Moving object to target position')
        self.mc.send_coords([XY_END[0], XY_END[1], HEIGHT_SAFE, 0, 180, 90], 15, 0)
        time.sleep(4)

        print('    Lowering object')
        self.mc.send_coords([XY_END[0], XY_END[1], HEIGHT_END, 0, 180, 90], 20, 0)
        time.sleep(3)

        # Turn off pump
        self.pump_off()

        print('    Returning to zero position')
        self.mc.send_angles([0, 0, 0, 0, 0, 0], 40)
        time.sleep(2)

    def set_led(self, r, g, b):
        """Set the LED color of the robot"""
        self.mc.set_color(r, g, b)

    def gripper_move(self, XY_START=[230, -50], HEIGHT_START=103, XY_END=[100, 220],
                     HEIGHT_END=108, HEIGHT_SAFE=220):
        """Move object using gripper from start to end position"""
        print('Starting gripper move sequence')

        # Set interpolation mode
        self.mc.set_fresh_mode(0)

        # Open gripper
        self.gripper_open()

        print('    Moving gripper above object')
        self.mc.send_coords([XY_START[0], XY_START[1], HEIGHT_SAFE, 0, 180, 90], 20, 0)
        time.sleep(2)

        print('    Lowering gripper to grab object')
        self.mc.send_coords([XY_START[0], XY_START[1], HEIGHT_START, 0, 180, 90], 15, 0)
        time.sleep(2)

        # Close gripper
        self.gripper_close()

        print('    Lifting object')
        self.mc.send_coords([XY_START[0], XY_START[1], HEIGHT_SAFE, 0, 180, 90], 15, 0)
        time.sleep(2)

        print('    Moving object to target position')
        self.mc.send_coords([XY_END[0], XY_END[1], HEIGHT_SAFE, 0, 180, 90], 15, 0)
        time.sleep(2)

        print('    Lowering object')
        self.mc.send_coords([XY_END[0], XY_END[1], HEIGHT_END, 0, 180, 90], 20, 0)
        time.sleep(2)

        # Open gripper
        self.gripper_open()

        print('    Lifting gripper')
        self.mc.send_coords([XY_END[0], XY_END[1], HEIGHT_SAFE, 0, 180, 90], 20, 0)

        print('    Returning to zero position')
        self.mc.send_angles([0, 0, 0, 0, 0, 0], 40)
        time.sleep(2)


if __name__ == '__main__':
    mc = MyCobot280Socket("192.168.31.175", 9001)
    camera_controller = CameraController("192.168.31.175", 8000)
    robot = RobotController(mc, camera_controller)

    # mc.release_all_servos()
    # robot.gripper_move()

    robot.back_zero()
    # robot.move_to_top_view()
    # mc.send_coords([82, -119, 103, 0, 180, 90], 15)
    # mc.send_coords([82, -119, 95, 0, 180, 90], 15)
    # mc.send_coords([82, -109, 90, 0, 180, 90], 10)
