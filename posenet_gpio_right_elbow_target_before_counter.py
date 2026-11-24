#!/usr/bin/python3

import argparse
import sys
import threading
import time

import Jetson.GPIO as GPIO
import jetson.inference
import jetson.utils
import numpy as np


parser = argparse.ArgumentParser(
    description="Use PoseNet to detect the right-elbow angle and control GPIO.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        jetson.inference.poseNet.Usage()
        + jetson.utils.videoSource.Usage()
        + jetson.utils.videoOutput.Usage()
        + jetson.utils.logUsage()
    ),
)

parser.add_argument(
    "input_URI", type=str, default="", nargs="?", help="URI of the input stream"
)
parser.add_argument(
    "output_URI", type=str, default="", nargs="?", help="URI of the output stream"
)
parser.add_argument(
    "--network", type=str, default="resnet18-body", help="PoseNet model"
)
parser.add_argument(
    "--overlay",
    type=str,
    default="links,keypoints",
    help="PoseNet overlay: links,keypoints,boxes,none",
)
parser.add_argument(
    "--threshold", type=float, default=0.15, help="minimum detection threshold"
)

try:
    opt = parser.parse_known_args()[0]
except Exception:
    print("")
    parser.print_help()
    sys.exit(0)


# Use physical BOARD numbering. Physical pin 13 is connected to relay IN2.
GPIO.setmode(GPIO.BOARD)
relay_pin = 13
GPIO.setup(relay_pin, GPIO.OUT)
GPIO.output(relay_pin, GPIO.LOW)


net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)
input_stream = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output_stream = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
font = jetson.utils.cudaFont()


def calculate_angle(pose, id1, id2, id3):
    """Calculate id1-id2-id3 in degrees, with id2 as the vertex."""
    point1 = np.array(
        [pose.Keypoints[id1].x, pose.Keypoints[id1].y], dtype=np.float32
    )
    point2 = np.array(
        [pose.Keypoints[id2].x, pose.Keypoints[id2].y], dtype=np.float32
    )
    point3 = np.array(
        [pose.Keypoints[id3].x, pose.Keypoints[id3].y], dtype=np.float32
    )

    vector1 = point1 - point2
    vector2 = point3 - point2
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    if denominator <= 1e-6:
        return None

    cosine_angle = np.dot(vector1, vector2) / denominator
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def select_target_pose(poses, img):
    """Select the largest person whose center is inside the central ROI."""
    target_pose = None
    largest_area = 0.0

    for pose in poses:
        if len(pose.Keypoints) < 3:
            continue

        x_values = [keypoint.x for keypoint in pose.Keypoints]
        y_values = [keypoint.y for keypoint in pose.Keypoints]

        left = min(x_values)
        right = max(x_values)
        top = min(y_values)
        bottom = max(y_values)

        pose_width = right - left
        pose_height = bottom - top
        pose_area = pose_width * pose_height

        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0

        # Keep people whose centers are inside the central part of the image.
        inside_center = (
            img.width * 0.2 < center_x < img.width * 0.8
            and img.height * 0.1 < center_y < img.height * 0.95
        )

        # Ignore small people in the background.
        large_enough = pose_area > img.width * img.height * 0.05

        if inside_center and large_enough and pose_area > largest_area:
            largest_area = pose_area
            target_pose = pose

    return target_pose


def blink_relay_twice():
    """Blink the relay twice without blocking the video thread."""
    for _ in range(2):
        GPIO.output(relay_pin, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(relay_pin, GPIO.LOW)
        time.sleep(0.5)


TRIGGER_ANGLE = 90.0
RESET_ANGLE = 80.0
angle_was_over_90 = False
relay_thread = None


try:
    while True:
        img = input_stream.Capture()

        if img is None:
            continue

        poses = net.Process(img, overlay=opt.overlay)
        print("Detected {:d} person(s)".format(len(poses)))

        frame_angles = []
        target_pose = select_target_pose(poses, img)

        if target_pose is None:
            poses_to_process = []
            print("No target person in the central area")
        else:
            poses_to_process = [target_pose]

        # Only calculate the selected target person's right-elbow angle.
        for pose in poses_to_process:
            shoulder_id = pose.FindKeypoint("right_shoulder")
            elbow_id = pose.FindKeypoint("right_elbow")
            wrist_id = pose.FindKeypoint("right_wrist")

            if shoulder_id < 0 or elbow_id < 0 or wrist_id < 0:
                continue

            angle = calculate_angle(pose, shoulder_id, elbow_id, wrist_id)

            if angle is None:
                continue

            frame_angles.append(angle)
            print("Right elbow angle: {:.2f} degrees".format(angle))

            elbow_x = int(pose.Keypoints[elbow_id].x)
            elbow_y = int(pose.Keypoints[elbow_id].y)

            font.OverlayText(
                img,
                text="{:.1f} deg".format(angle),
                x=elbow_x + 10,
                y=elbow_y - 10,
                color=font.White,
                background=font.Gray40,
            )

        if frame_angles:
            max_angle = max(frame_angles)
            relay_is_running = relay_thread is not None and relay_thread.is_alive()

            if (
                max_angle > TRIGGER_ANGLE
                and not angle_was_over_90
                and not relay_is_running
            ):
                print(
                    "Right elbow angle is greater than "
                    "{} degrees - relay activated".format(TRIGGER_ANGLE)
                )

                angle_was_over_90 = True
                relay_thread = threading.Thread(
                    target=blink_relay_twice,
                    daemon=True,
                )
                relay_thread.start()

            elif max_angle < RESET_ANGLE and not relay_is_running:
                GPIO.output(relay_pin, GPIO.LOW)
                angle_was_over_90 = False

        output_stream.Render(img)
        output_stream.SetStatus(
            "{} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS())
        )

        net.PrintProfilerTimes()

        if not input_stream.IsStreaming() or not output_stream.IsStreaming():
            break

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    if relay_thread is not None and relay_thread.is_alive():
        relay_thread.join()

    GPIO.output(relay_pin, GPIO.LOW)
    GPIO.cleanup()
    print("GPIO cleaned up.")
