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
import os

from flask import Flask, render_template, Response

from facenet import take_images, align_raw_data, generate_classifier
from camera import VideoCamera
from facenet.realtime_face_recognition import RecognitionCamera

app = Flask(__name__)

file_dir_path, _ = os.path.split(__file__)
print(file_dir_path)

@app.route('/watch')
def index():
    print('hi')
    return render_template('index.html')


@app.route('/take_pictures')
def take_pictures():
    take_images.save_raw_images('tester22')
    print('doneeeeeeeeee')
    return render_template('generated_classifier.html')


@app.route('/generate_classifier')
def generate_classifier():
    print('generating...')

    align_data_dir = os.path.join(file_dir_path, '/aligned_data')
    model_path = os.path.join(file_dir_path, 'assets/model_VGGFace2_Inception-ResNet-v1/20180402-114759')
    print(file_dir_path)
    classifier_path = os.path.join(file_dir_path, '/assets/classifier_first.pkl')

    align_raw_data.align_raw_images()
    generate_classifier.generate_classifier(mode='TRAIN', data_dir=align_data_dir,
                                            model=model_path,
                                            classifier_path=classifier_path, batch_size=1000,
                                            min_nrof_images_per_class=10, nrof_train_images_per_class=15,
                                            use_split_dataset=True)

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
