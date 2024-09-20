import pandas as pdb
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.layers import MaxPooling2D, Input
from keras.src.models.sequential import Sequential
from keras.src.layers.reshaping.flatten import Flatten
from keras.src.layers.core.dense import Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.src.layers.normalization.batch_normalization import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam
import cv2
import time
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from scipy.signal._wavelets import cascade
from pickle import NONE
from keras.src.applications.resnet import ResNet50
from keras.src.layers.pooling.global_average_pooling2d import GlobalAveragePooling2D
from keras.src.models.model import Model
from sklearn.utils.class_weight import compute_class_weight
from keras.regularizers import l2
from sklearn.utils import class_weight

model = load_model('emotionRecognitionModel.keras')


camera = cv2.VideoCapture(0)

emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'suprise', 6: 'neutral'}
imageList = []
emotionList = []

cascadeFace = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



if not camera.isOpened():
    print("Camera is not working")
    exit()

def processFace(face):
    greyFrame = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    faceDetection = cascadeFace.detectMultiScale(greyFrame, scaleFactor = 1.1, minNeighbors=5, minSize=(30,30))

    if len(faceDetection) == 0:
        print("No face detected")
        return None

    (x, y, w, h) = faceDetection[0]
    faces = face[y:y+h, x:x+w]
    greyscaleFace = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
    sharpenedFace = sharpenImage(greyscaleFace)
    resize = cv2.resize(sharpenedFace, (48,48))
    faceNormalisation = resize/ 255.0
    reshape = np.reshape(faceNormalisation, (1, 48, 48, 1))
    return reshape

def prediction(model, array):
    predictions = model.predict(array)
    predictedClasses = np.argmax(predictions, axis=1)
    return predictedClasses

def sharpenImage(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened
#
while True:
    ret, frame = camera.read()

    if not ret:
        print("Can not recieve frame. Exiting")
        break

    imageArray = processFace(frame)

    if imageArray is not None:
        index = prediction(model, imageArray)

        emotion = emotions.get(index[0], 'unclear')

        cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

    
dataset = pdb.read_csv('train.csv')

imageDictionary = {emotion: [] for emotion in emotions.values()}

for index, row in dataset.iterrows():
    # row[] retrieves the value from current row specified by heading
    emotion = emotions[row['emotion']]
    pixels = row['pixels']
    image = np.fromstring(pixels, sep=' ', dtype=float).reshape(48,48)
    
    imageDictionary[emotion].append(image)
    
for emotion in imageDictionary:
    imageDictionary[emotion] = np.array(imageDictionary[emotion])


for emotion, image in imageDictionary.items():
    imageList.extend(image)
    emotionList.extend([emotion] * len(image))
    
imageList = np.array(imageList)
emotionList = np.array(emotionList)

imageList = imageList / 255.0

labelEncoder = LabelEncoder()
emotionListEncoded = labelEncoder.fit_transform(emotionList)
emotionListHotEncode = to_categorical(emotionListEncoded)

imageList = np.expand_dims(imageList, axis=-1)

trainImageList, imageListValidation, trainEmotionList, emotionListValidation = train_test_split(imageList, emotionListHotEncode, test_size=0.2, random_state=42)


class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(trainEmotionList), y = trainEmotionList.ravel())
class_weights_dict = dict(enumerate(class_weights))

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(len(labelEncoder.classes_), activation='softmax')
])

dataaug = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.06,
    height_shift_range=0.06,
    shear_range=0.06,
    zoom_range=0.06,
    horizontal_flip=True,
    fill_mode='nearest'
    
)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

earlyStop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(dataaug.flow(trainImageList, trainEmotionList, batch_size=16),
                    epochs=60, 
                    validation_data=(imageListValidation, emotionListValidation),
                    class_weight=class_weights_dict,
                    callbacks=[earlyStop],
                    verbose=1)
              

loss, accuracy = model.evaluate(trainImageList, trainEmotionList, verbose=1)
print(f'Validation accuracy: {accuracy:.4f}')

model.save('emotionRecognitionModel.keras')