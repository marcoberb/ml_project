# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import sys
import time
import urllib
import cv2
import numpy as np
import logging
import face

# IP_CAM
global url_ipcam
url_ipcam = "http://10.25.254.165:8080/shot.jpg"  # change this when ipcam app is on another URL


# returns image retrieved from url
def get_img_from_ipcam(url):

    is_img_retrieved = True
    img_retrieved = None

    try:
        img_response = urllib.request.urlopen(url)  # retrieve current frame
        img_np = np.array(bytearray(img_response.read()))
        img_retrieved = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)  # frame to be displayed
        img_retrieved
    except Exception as e:
        logging.debug(e)
        is_img_retrieved = False

    return is_img_retrieved, img_retrieved


def add_overlays(frame, faces, frame_rate):

    if faces is not None:

        for face in faces:

            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)


def main(args):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0

    # video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()
    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True

    while True:

        # Capture frame-by-frame
        # ret, frame = video_capture.read()

        ret, frame = get_img_from_ipcam(url_ipcam)

        if ret:

            if (frame_count % frame_interval) == 0:
                faces = face_recognition.identify(frame)

                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

            add_overlays(frame, faces, frame_rate)

            frame_count += 1
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video', 1200, 900)
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("error retrieving img, ret = ", ret)

    # When everything is done, release the capture
    # video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', help='Enable some debug outputs.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
