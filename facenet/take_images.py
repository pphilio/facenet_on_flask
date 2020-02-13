import os
import time

import cv2
import face_recognition
import numpy as np

file_dir_path, _ = os.path.split(__file__)


def __current_time():
    return time.strftime("%Y%M%D_%H%M%S", time.localtime())


def save_raw_images(name=__current_time()):
    known_face_encodings = []
    known_face_names = []
    face_names = []

    cnt = 0
    video_capture = cv2.VideoCapture(0)
    print('start taking images')

    output_path = os.path.join(file_dir_path, './raw_images/{}'.format(name))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    while cnt < 300:
        # grab a single frame of video
        ret, frame = video_capture.read()
        print('1')
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        print('11')
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
            if True not in matches:
                known_face_names.append(name)
                known_face_encodings.append(face_encodings[0])
            print(face_encoding)
            print(known_face_encodings)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            print("Took images..: {}".format(cnt / 10))
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if cnt % 10 == 0:
                    output_file = '{}/{}.jpg'.format(output_path, int(cnt / 10))
                    print("file saved to '{}'".format(output_file))

                    cv2.imwrite(output_file, frame)
                    ret, jpeg = cv2.imencode('.jpg', frame)
                cnt += 1
            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.destroyAllWindows()
    print('done')
    return 0