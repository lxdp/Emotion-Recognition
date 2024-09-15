import pandas as pdb
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.layers import MaxPooling2D
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

model = load_model('emotionRecognitionModel.keras')


camera = cv2.VideoCapture(0)

emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'suprise', 6: 'neutral'}
imageList = []
emotionList = []



if not camera.isOpened():
    print("Camera is not working")
    exit()
    
def processFace(face):
    greyscaleFace = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(greyscaleFace, (48,48))
    faceNormalisation = resize/ 255.0
    reshape = np.reshape(faceNormalisation, (1, 48, 48, 1))
    return reshape
    
def prediction(model, array):
    predictions = model.predict(array)
    predictedClasses = np.argmax(predictions, axis=1)
    return predictedClasses
    
while True:
    ret, frame = camera.read()
    
    if not ret:
        print("Can not recieve frame. Exiting")
        break
    
    imageArray = processFace(frame)
    
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


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labelEncoder.classes_), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

earlyStop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

history = model.fit(trainImageList, trainEmotionList,
                    batch_size=16,
                    epochs=40, 
                    validation_data=(imageListValidation, emotionListValidation),
                    callbacks=[earlyStop],
                    verbose=1)

loss, accuracy = model.evaluate(trainImageList, trainEmotionList, verbose=1)
print(f'Validation accuracy: {accuracy:.4f}')

model.save('emotionRecognitionModel.keras')