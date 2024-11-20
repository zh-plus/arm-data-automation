import queue
import threading
import time

import cv2
import gradio as gr

from grounded_sam2.grounded_sam2_florence2_open_vocal_detection import open_vocab_detection


class EnhancedCameraWebUI:
    def __init__(self, camera_controller, data_automation=None):
        self.camera = camera_controller
        self.data_automation = data_automation
        self.is_streaming = False
        self.frame_buffer = queue.Queue(maxsize=5)
        self.streaming_thread = None
        self.skip_confirmation = False

        # Robot automation state
        self.is_automation_running = False
        self.current_status = "Ready"
        self.movement_queue = queue.Queue()
        self.current_preview = None
        self.running_command = 'No active command'
        self.step_info = 'Step 0: Ready'
        self.current_step = 0

        # Add collected data storage
        self.collected_data = []
        self.automation_id = 0

    def _stream_worker(self):
        """Background worker to continuously fetch frames."""
        while self.is_streaming:
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    try:
                        self.frame_buffer.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.frame_buffer.get_nowait()
                            self.frame_buffer.put_nowait(frame)
                        except queue.Empty:
                            pass
            except Exception as e:
                print(f"Streaming worker error: {e}")
                time.sleep(0.1)
                self.toggle_stream()
                time.sleep(0.1)
                self.toggle_stream()

    def get_stream_frame(self):
        try:
            if self.is_streaming:
                try:
                    frame = self.frame_buffer.get_nowait()
                    return frame
                except queue.Empty:
                    return None
            return None
        except Exception as e:
            print(f"Streaming error: {e}")
            return None

    def toggle_stream(self):
        try:
            if not self.is_streaming:
                success = self.camera.start_stream()
                if success:
                    self.is_streaming = True
                    while not self.frame_buffer.empty():
                        self.frame_buffer.get_nowait()
                    self.streaming_thread = threading.Thread(
                        target=self._stream_worker,
                        daemon=True
                    )
                    self.streaming_thread.start()
                    return "Camera streaming started"
                return "Failed to start camera stream"
            else:
                self.is_streaming = False
                if self.streaming_thread:
                    self.streaming_thread.join(timeout=1.0)
                self.camera.stop_stream()
                return "Camera streaming stopped"
        except Exception as e:
            return f"Stream toggle error: {str(e)}"

    def get_collected_data_as_list(self):
        """Convert collected data to list format for Gradio Dataframe."""
        return [[d["id"], d["command"], d["status"]] for d in self.collected_data]

    def start_automation(self):
        """Start the robot automation process."""
        if not self.data_automation:
            return [None, self.get_collected_data_as_list()]

        if not self.is_automation_running:
            self.is_automation_running = True
            self.current_step = 0
            self.step_info = 'Step 0: Ready'
            thread = threading.Thread(
                target=self._automation_worker,
                daemon=True
            )
            thread.start()
            time.sleep(0.1)
            self.current_status = "Automation started"
            return [None, self.get_collected_data_as_list()]
        return [self.current_preview, self.get_collected_data_as_list()]

    def stop_automation(self):
        """Stop the robot automation process."""
        self.is_automation_running = False
        self.current_status = "Automation stopped"
        return [None, self.get_collected_data_as_list()]

    def reset_automation(self):
        """Reset the robot automation state."""
        self.is_automation_running = False
        self.current_step = 0
        self.step_info = 'Step 0: Ready'
        self.current_preview = None
        self.current_status = "System reset"
        self.collected_data = []  # Clear collected data
        return [None, self.get_collected_data_as_list()]

    def confirm_movement(self):
        """Confirm the current movement step."""
        self.movement_queue.put(True)
        return [None, self.get_collected_data_as_list()]

    def cancel_movement(self):
        """Cancel the current movement step."""
        self.movement_queue.put(False)
        return [None, self.get_collected_data_as_list()]

    def toggle_skip_confirmation(self, value):
        """Toggle the skip confirmation setting."""
        self.skip_confirmation = value
        return [
            f"Skip confirmation {'enabled' if value else 'disabled'}",
            gr.update(interactive=not value and not self.is_automation_running),
            gr.update(interactive=not value and not self.is_automation_running)
        ]

    def _automation_worker(self):
        """Main automation worker thread."""
        try:
            for _ in range(2):
                for command_pair in self.data_automation.command_pair:
                    if not self.is_automation_running:
                        break

                    for command in command_pair:
                        if not self.is_automation_running:
                            break

                        success = False
                        for i in range(1, 4):
                            self.current_status = f"retrying {command}: {i}-th time"
                            success = self._execute_command_with_gui(command)

                            if success:
                                break
                        if not success:
                            self.is_automation_running = False
                            self.current_status = "Automation stopped due to error"
                            break
            self.step_info = "END"
            self.current_status = "END"
            self.data_automation.robot.back_zero()

        except Exception as e:
            self.current_status = f"Error: {str(e)}"
            self.is_automation_running = False
            print(e)

    def _execute_command_with_gui(self, command):
        """Execute a single automation command with GUI updates."""
        try:
            self.running_command = command
            self.automation_id += 1
            current_id = self.automation_id

            # Step 1: Move robot to zero position
            self.current_step = 1
            self.step_info = f'Step 1: Robot Init'
            self.current_status = "Moving robot to zero position"
            self.data_automation.robot.back_zero()

            # Step 2: Process instruction
            self.current_step = 2
            self.step_info = 'Step 2: Camera Init'
            self.current_status = f"Moving camera to top view"

            if self.camera.is_streaming():
                self.toggle_stream()
                image_path = self.data_automation.robot.top_view_shot()
                self.toggle_stream()
            else:
                image_path = self.data_automation.robot.top_view_shot()

            if not image_path or "Error" in image_path:
                raise RuntimeError(f"Failed to capture image: {image_path}")

            # Step 3: Get detection result
            self.current_step = 3
            self.step_info = 'Step 3: Command analysis'
            self.current_status = f"Asking VLM the image and {command}: What should I do?"
            result = self.data_automation.vlm_agent.detect_names(
                f'Instruction:\n{command}',
                image_path
            )
            names = [result['start'], result['end']]

            # Step 4: Get object positions and create visualization
            self.current_step = 4
            self.step_info = 'Step 4: Object detection'
            self.current_status = f"Running open vocabulary detection using {names[0]} and {names[1]}"

            detection_results, detection_viz_paths = open_vocab_detection(
                image_path,
                names
            )

            start_x, start_y, end_x, end_y = self.data_automation.get_detected_position(
                detection_results,
                names
            )

            self.current_status = 'Visualizing results...'
            # Create movement visualization
            masked_image = cv2.imread(str(detection_viz_paths['masks']))
            cv2.arrowedLine(
                masked_image,
                (int(start_x), int(start_y)),
                (int(end_x), int(end_y)),
                color=(0, 255, 0),
                thickness=2,
                tipLength=0.3
            )
            self.current_preview = masked_image

            if not self.skip_confirmation:
                # Wait for user confirmation
                self.current_status = "Waiting for movement confirmation"
                confirmed = self.movement_queue.get()

                if not confirmed:
                    self.collected_data.append({
                        "id": current_id,
                        "command": command,
                        "status": "Cancelled"
                    })
                    return False
            else:
                self.current_status = "Skipping confirmation (auto-confirmed)"

            # Execute movement
            self.current_step = 5
            self.step_info = 'Step 5: Action'
            self.current_status = "Moving and grasping"
            start_x_robot, start_y_robot = self.data_automation.robot.eye2hand(
                start_x, start_y
            )
            end_x_robot, end_y_robot = self.data_automation.robot.eye2hand(
                end_x, end_y
            )

            self.data_automation.robot.gripper_move(
                XY_START=[start_x_robot, start_y_robot],
                XY_END=[end_x_robot, end_y_robot]
            )

            # Check completion
            self.current_step = 6
            self.step_info = 'Step 6: Completeness Verification'
            self.current_status = f"Asking VLM with the image: Have I done my task ({self.running_command})?"
            if self.camera.is_streaming():
                self.toggle_stream()
                image_path = self.data_automation.robot.top_view_shot()
                self.toggle_stream()
            else:
                image_path = self.data_automation.robot.top_view_shot()

            success = self.data_automation.judge_result(
                result['description'],
                command,
                image_path
            )

            # Record the data
            self.collected_data.append({
                "id": current_id,
                "command": command,
                "status": "Success" if success else "Failed"
            })

            self.current_status = "Task completed successfully" if success else "Task failed"
            self.current_preview = None

            return success

        except Exception as e:
            self.collected_data.append({
                "id": current_id,
                "command": command,
                "status": f"Error: {str(e)}"
            })
            self.current_status = f"Error: {str(e)}"
            self.current_preview = None
            print(str(e))
            return False

    def create_interface(self):
        """Create and configure the Gradio interface."""
        with gr.Blocks(title="Enhanced Camera Control with Robot Automation") as interface:
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        # Camera controls
                        stream_btn = gr.Checkbox(
                            label="Enable Camera Stream",
                            value=False,
                            interactive=True
                        )

                        skip_confirm = gr.Checkbox(
                            label="Skip Confirmation",
                            value=False,
                            interactive=True
                        )

                    # Camera feed
                    camera_feed = gr.Image(
                        value=self.get_stream_frame,
                        label="Camera Feed",
                        type='numpy',
                        every=0.033,
                        interactive=False,
                        streaming=True
                    )

                    # Automation controls
                    with gr.Row():
                        start_btn = gr.Button("Start Automation", variant="primary")
                        stop_btn = gr.Button("Stop Automation", variant="secondary")
                        reset_btn = gr.Button("Reset Automation", variant="secondary")

                    # Movement preview
                    preview_image = gr.Image(
                        label="Movement Preview",
                        interactive=False,
                        value=lambda: self.current_preview if self.current_preview is not None else None,
                        every=0.5
                    )

                    # Movement confirmation
                    with gr.Row():
                        confirm_btn = gr.Button("Confirm Movement", variant="primary")
                        cancel_btn = gr.Button("Cancel Movement", variant="secondary")

                with gr.Column(scale=1):
                    # Status Information Panel
                    with gr.Group():
                        gr.Label(value="Status Information")
                        current_command = gr.Textbox(
                            label="Current Command",
                            value=lambda: self.running_command,
                            interactive=False,
                            every=1
                        )
                        automation_step = gr.Textbox(
                            label="Automation Step",
                            value=lambda: self.step_info,
                            interactive=False,
                            every=1
                        )
                        info_status = gr.Textbox(
                            label="Info",
                            value=lambda: self.current_status,
                            interactive=False,
                            every=1
                        )

                    # Add Collected Data table
                    with gr.Group():
                        gr.Label(value="Collected Data")
                        collected_data_table = gr.Dataframe(
                            headers=["ID", "Command", "Status"],
                            datatype=["number", "str", "str"],
                            value=self.get_collected_data_as_list,
                            every=1,
                            label="Automation History"
                        )

            # Event handlers with corrected outputs
            start_btn.click(
                fn=self.start_automation,
                outputs=[preview_image, collected_data_table]
            )

            stop_btn.click(
                fn=self.stop_automation,
                outputs=[preview_image, collected_data_table]
            )

            reset_btn.click(
                fn=self.reset_automation,
                outputs=[preview_image, collected_data_table]
            )

            confirm_btn.click(
                fn=self.confirm_movement,
                outputs=[preview_image, collected_data_table]
            )

            cancel_btn.click(
                fn=self.cancel_movement,
                outputs=[preview_image, collected_data_table]
            )

            # Stream toggle event
            stream_btn.change(
                fn=self.toggle_stream,
                outputs=[info_status]
            )

            skip_confirm.change(
                fn=self.toggle_skip_confirmation,
                inputs=skip_confirm,
                outputs=[info_status, confirm_btn, cancel_btn]
            )

            # Interface state updates
            def update_ui_state():
                button_interactive = not self.is_automation_running and not self.skip_confirmation
                return {
                    stream_btn: gr.update(interactive=not self.is_automation_running),
                    skip_confirm: gr.update(interactive=not self.is_automation_running),
                    confirm_btn: gr.update(interactive=button_interactive),
                    cancel_btn: gr.update(interactive=button_interactive),
                    current_command: gr.update(value=self.running_command),
                    automation_step: gr.update(value=self.step_info),
                    info_status: gr.update(value=self.current_status),
                    collected_data_table: gr.update(value=self.get_collected_data_as_list())
                }

            # Update interface state
            interface.load(
                fn=update_ui_state,
                outputs=[
                    stream_btn,
                    skip_confirm,
                    confirm_btn,
                    cancel_btn,
                    current_command,
                    automation_step,
                    info_status,
                    collected_data_table
                ]
            )

            return interface

    def run(self):
        """Run the web interface."""
        interface = self.create_interface()
        interface.queue(default_concurrency_limit=5).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
