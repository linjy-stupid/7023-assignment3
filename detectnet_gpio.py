#!/usr/bin/python3
#
# PoseNet + GPIO:
# Blink the relay twice when the right-elbow angle becomes greater than 90 degrees.
#

import argparse
import sys
import time

import Jetson.GPIO as GPIO
import jetson.inference
import jetson.utils
import numpy as np


parser = argparse.ArgumentParser(
    description="React through GPIO when the right-elbow angle is greater than 90 degrees.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=(
        jetson.inference.poseNet.Usage()
        + jetson.utils.videoSource.Usage()
        + jetson.utils.videoOutput.Usage()
        + jetson.utils.logUsage()
    ),
)

parser.add_argument("input_URI", type=str, default="", nargs="?", help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs="?", help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pose model to load")
parser.add_argument(
    "--overlay",
    type=str,
    default="links,keypoints",
    help="pose overlay flags: links, keypoints, boxes, none",
)
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold")

try:
    opt = parser.parse_known_args()[0]
except Exception:
    print("")
    parser.print_help()
    sys.exit(0)


# Use physical BOARD numbering.  Physical pin 13 is connected to relay IN2.
GPIO.setmode(GPIO.BOARD)
relay_pin = 13
GPIO.setup(relay_pin, GPIO.OUT)
GPIO.output(relay_pin, GPIO.LOW)

net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)
input_stream = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output_stream = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
font = jetson.utils.cudaFont()


def calculate_angle(pose, id1, id2, id3):
    """Calculate id1-id2-id3 in degrees, with id2 as the angle vertex."""
    point1 = np.array([pose.Keypoints[id1].x, pose.Keypoints[id1].y])
    point2 = np.array([pose.Keypoints[id2].x, pose.Keypoints[id2].y])
    point3 = np.array([pose.Keypoints[id3].x, pose.Keypoints[id3].y])

    vector1 = point1 - point2
    vector2 = point3 - point2
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)

    if denominator == 0:
        return None

    cosine_angle = np.dot(vector1, vector2) / denominator
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def blink_relay_twice():
    """Turn the relay on and off twice."""
    for _ in range(2):
        GPIO.output(relay_pin, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(relay_pin, GPIO.LOW)
        time.sleep(0.5)


# Trigger once after crossing 90 degrees, then re-arm at 90 degrees or below.
angle_was_over_90 = False

try:
    while True:
        img = input_stream.Capture()

        if img is None:
            continue

        poses = net.Process(img, overlay=opt.overlay)
        print("detected {:d} person(s) in image".format(len(poses)))

        valid_angle_found = False
        angle_is_over_90 = False

        for pose in poses:
            shoulder_id = pose.FindKeypoint("right_shoulder")
            elbow_id = pose.FindKeypoint("right_elbow")
            wrist_id = pose.FindKeypoint("right_wrist")

            if shoulder_id < 0 or elbow_id < 0 or wrist_id < 0:
                continue

            angle = calculate_angle(pose, shoulder_id, elbow_id, wrist_id)

            if angle is None:
                continue

            valid_angle_found = True
            print("Right elbow angle: {:.2f} degrees".format(angle))

            # Draw the value beside the right-elbow keypoint.
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

            if angle > 90.0:
                angle_is_over_90 = True

        if angle_is_over_90 and not angle_was_over_90:
            print("Right elbow angle is greater than 90 degrees - relay activated")
            blink_relay_twice()
            angle_was_over_90 = True
        elif valid_angle_found and not angle_is_over_90:
            GPIO.output(relay_pin, GPIO.LOW)
            angle_was_over_90 = False

        output_stream.Render(img)
        output_stream.SetStatus(
            "{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS())
        )

        net.PrintProfilerTimes()

        if not input_stream.IsStreaming() or not output_stream.IsStreaming():
            break
finally:
    GPIO.output(relay_pin, GPIO.LOW)
    GPIO.cleanup()
