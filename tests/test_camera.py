import time

import cv2
from camera import CameraController


def test_camera_connection():
    """Test basic camera connection and operations"""
    # Initialize camera controller
    camera = CameraController(host='192.168.31.175', port=8000)
    camera.start_stream()

    try:
        print("\n=== Testing Camera Operations ===")

        # Test 1: Get a single frame
        print("\nTest 1: Getting single frame...")
        frame = camera.get_frame()
        if frame is not None:
            print("✓ Successfully got frame")
            # Display frame
            cv2.imshow('Test Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        else:
            print("✗ Failed to get frame")

        # Test 2: Save image
        print("\nTest 2: Saving image...")
        result = camera.save_image()
        print(f"Save result: {result}")

        # Test 3: Get camera properties
        print("\nTest 3: Getting camera properties...")
        properties = ['brightness', 'contrast', 'saturation']
        for prop in properties:
            value = camera.get_camera_property(prop)
            print(f"{prop}: {value}")

        # Test 4: Set camera properties
        print("\nTest 4: Setting camera properties...")
        test_values = {
            'brightness': 50,
            'contrast': 60,
            'saturation': 70,
        }

        for prop, value in test_values.items():
            print(f"\nSetting {prop} to {value}")
            result = camera.set_camera_property(prop, value)
            print(f"Set result: {result}")

            # Verify the change
            new_value = camera.get_camera_property(prop)
            print(f"New {prop} value: {new_value}")

        # Test 5: Continuous frame capture
        print("\nTest 5: Testing continuous frame capture (5 seconds)...")
        start_time = time.time()
        frames_received = 0

        while time.time() - start_time < 5:
            frame = camera.get_frame()
            if frame is not None:
                frames_received += 1
                cv2.imshow('Continuous Test', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        fps = frames_received / 5
        print(f"Received {frames_received} frames in 5 seconds (approximately {fps:.2f} FPS)")

        # Test 6: Error handling
        print("\nTest 6: Testing error handling...")
        # Test invalid property
        print("Testing invalid property...")
        result = camera.get_camera_property('invalid_property')
        print(f"Invalid property result: {result}")

        # Test invalid property value
        print("\nTesting invalid property value...")
        result = camera.set_camera_property('brightness', -999)
        print(f"Invalid value result: {result}")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
    finally:
        cv2.destroyAllWindows()


def test_performance():
    """Test camera performance metrics"""
    camera = CameraController(host='192.168.31.175', port=8000)

    print("\n=== Testing Performance ===")

    # Test frame retrieval speed
    print("\nMeasuring frame retrieval speed...")
    times = []
    frames = 50

    for i in range(frames):
        start = time.time()
        frame = camera.get_frame()
        if frame is not None:
            times.append(time.time() - start)
            print(f"Frame {i + 1}/{frames}", end='\r')

    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage frame retrieval time: {avg_time:.3f} seconds")
        print(f"Approximate max FPS: {1 / avg_time:.2f}")


def test_reliability():
    """Test camera reliability with rapid requests"""
    camera = CameraController(host='192.168.31.175', port=8000)

    print("\n=== Testing Reliability ===")

    # Test rapid property changes
    print("\nTesting rapid property changes...")
    test_values = [0, 50, 100, 75, 25]

    for value in test_values:
        result = camera.set_camera_property('brightness', value)
        print(f"Set brightness to {value}: {result}")
        time.sleep(0.1)  # Small delay


if __name__ == "__main__":
    print("Starting Camera Client Tests...")

    try:
        # Run all tests
        test_camera_connection()
        test_performance()
        test_reliability()

        print("\n✓ All tests completed!")

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\n✗ Tests failed with error: {e}")
    finally:
        cv2.destroyAllWindows()
