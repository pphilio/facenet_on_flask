import os
import time

import cv2

from facenet import align_raw_data, generate_classifier, take_images
from facenet.contributed import face
from facenet.realtime_face_recognition import add_overlays

file_dir_path, _ = os.path.split(__file__)

align_data_dir = os.path.join(file_dir_path, './aligned_data')
model_path = os.path.join(file_dir_path, './assets/model_VGGFace2_Inception-ResNet-v1/20180402-114759')
classifier_path = os.path.join(file_dir_path, './assets/classifier_first.pkl')


def align_and_generate_classifier():
    align_raw_data.align_raw_images()

    generate_classifier.generate_classifier(mode='TRAIN', data_dir=align_data_dir,
                                            model=model_path,
                                            classifier_path=classifier_path, batch_size=1000,
                                            min_nrof_images_per_class=10, nrof_train_images_per_class=15,
                                            use_split_dataset=True)


def test_realtime_recognition(debug=False):
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition(model=model_path,
                                        classifier=classifier_path)
    start_time = time.time()

    if debug is True:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Check outr current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate)

        frame_count += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # align_and_generate_classifier()
    #
    # test_realtime_recognition(True)

    take_images.save_raw_images('tester')
