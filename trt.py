# from utils.utils import preproc, vis
import rclpy
from utils.utils import BaseEngine, MinimalPublisher, Predictor
import numpy as np
import cv2
import time
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-c", "--conf", type=float, default=0.1, help="confidence threshold")
    parser.add_argument("-v", "--video", type=int, help="camera index ")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="use nms")

    args = parser.parse_args()
    print(args)

    engine_path = args.engine
    camera_idx = args.video
    conf = args.conf
    end2end = args.end2end

    print("Starting video capture...\n")
    cap = cv2.VideoCapture(camera_idx)

    rclpy.init()
    minimal_publisher = MinimalPublisher(engine_path, conf, end2end, cap)
    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly and turn off video capture
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    cap.release()
    rclpy.shutdown()
