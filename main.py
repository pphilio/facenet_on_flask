#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response
from camera import VideoCamera
from facenet.realtime_face_recognition import recognize_realtime, RecognitionCamera
from facenet.test import align_and_generate_classifier

app = Flask(__name__)


@app.route('/watch')
def index():
    print('hi')
    return render_template('index.html')


@app.route('/generate_classifier')
def generate_classifier():
    print('generating...')
    align_and_generate_classifier()
    return render_template('generated_classifier.html')


@app.route('/realtime_recognize')
def realtime_recognize():
    print('recognizing...')
    return render_template('watch_realtime.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/realtime_recognition_feed')
def realtime_recognition_feed():
    return Response(gen(RecognitionCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
