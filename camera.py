import cv2


class VideoCamera(object):

    def __init__(self):
        self.frame_interval = 3  # Number of frames after which to run...
        self.fps_display_interval = 5  # seconds
        self.frame_rate = 0
        self.frame_count = 0
        self.test_str = "SangWon-Park♥"

        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video_capture = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video_capture.release()

    def get_frame(self):
        ret, frame = self.video_capture.read()


        # frame layout
        cv2.putText(frame, self.test_str, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)

        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
