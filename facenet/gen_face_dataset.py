import face_recognition
import cv2
import numpy as np
import os
from scipy import misc

known_face_encodings=[]
known_face_names=[]
face_names=[]
index=0
#todo 호출될때마다 인덱스 1씩 증가
cnt=0
video_capture = cv2.VideoCapture(0)
print('start')

while cnt<=300:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    output_path = './dataset/{}'.format(index)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        # matches=knownfaceencodings의 몇번째와 같은지 비교
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)  # tolerance 있음 lower=strict
        name = "Unknown"
        if not True in matches:
            known_face_names.append(name)
            known_face_encodings.append(face_encoding)

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        print(cnt%10)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if cnt%10==0:
                print('{}/{}.jpg'.format(output_path,int(cnt/10)))
                cv2.imwrite('{}/{}.jpg'.format(output_path, int(cnt/10)), frame)
            cnt = cnt + 1
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

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
