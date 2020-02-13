import sys
import time
import cv2

from facenet.contributed import face


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

def recognize_realtime(debug=False, frame_interval = 3, fps_display_interval = 5):
    frame_rate = 0
    frame_count = 0
    video_capture = cv2.VideoCaputre(0)
    face_recognition = face.Recognition()
    start_time = time.time()

    if debug is True:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if frame_count % frame_interval == 0:
            faces = face_recognition.identify(frame)

            # Check our current fps
            end_time= time.time()
            if end_time - start_time > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate)

        frame_count += 1
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

class RecognitionCamera(object):

    def __init__(self):
        self.frame_interval = 3
        self.fps_display_interval = 5
        self.frame_rate = 0
        self.frame_count = 0

        self.video_capture = cv2.VideoCapture(0)

    def __del__(self):
        self.video_capture.release()

    def get_frame(self):
        ret, frame = self.video_capture.read()

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

