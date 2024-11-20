from pymycobot import MyCobot280Socket

from auto_data import DataAutomation
from camera import CameraController
from gui import EnhancedCameraWebUI
from robot import RobotController

if __name__ == '__main__':
    # 默认使用9000端口
    # 其中"192.168.11.15"为机械臂IP，请自行输入你的机械臂IP

    mc = MyCobot280Socket("192.168.31.175", 9001)
    camera_controller = CameraController("192.168.31.175", 8000)

    robot = RobotController(mc, camera_controller)

    data_automation = DataAutomation(robot)

    gui = EnhancedCameraWebUI(camera_controller, data_automation)
    gui.run()
