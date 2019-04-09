import cv2
from keras.models import load_model
import numpy as np
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
from statistics import mode

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
labels = get_labels('fer2013')

#hyper parameters for counding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

#loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
model = load_model(emotion_model_path)

#getting input model shapes for inference
target_size = model.input_shape[1:3]

#starting lists for calculating modes
window = []

#starting video streaming
cv2.namedWindow('Window_Frame')
video_capture = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    RET, BRG_IMAGE = cap.read()

    BRG_IMAGE = np.flip(BRG_IMAGE, axis=1)

    gray_image = cv2.cvtColor(BRG_IMAGE, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(BRG_IMAGE, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        prediction = model.predict(gray_face)
        probability = np.max(prediction)
        label_arg = np.argmax(prediction)
        text = labels[label_arg]
        window.append(text)

        if len(window) > frame_window:
            window.pop(0)
        try:
            emode = mode(window)
        except:
            continue

        if text == 'angry':
            color = probability * np.asarray((255, 0, 0))
        elif text == 'sad':
            color = probability * np.asarray((0, 0, 255))
        elif text == 'happy':
            color = probability * np.asarray((255, 255, 0))
        elif text == 'surprise':
            color = probability * np.asarray((0, 255, 255))
        else:
            color = probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emode, color, 0, -45, 1,1)


    BRG_IMAGE = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Window_Frame', BRG_IMAGE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
