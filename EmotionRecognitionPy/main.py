from PIL import Image
import pandas as pdb
import numpy as np
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
from keras.src.legacy.preprocessing.image import ImageDataGenerator


dataset = pdb.read_csv('train.csv')

emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'suprise', 6: 'neutral'}
imageList = []
emotionList = []

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
    MaxPooling2D((2,2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(labelEncoder.classes_), activation='softmax')
])

dataaug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

earlyStop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(dataaug.flow(trainImageList, trainEmotionList, batch_size=32),
                    epochs=40, 
                    validation_data=(imageListValidation, emotionListValidation),
                    callbacks=[earlyStop],
                    verbose=1)

loss, accuracy = model.evaluate(trainImageList, trainEmotionList, verbose=1)
print(f'Validation accuracy: {accuracy:.4f}')

model.save('emotionRecognitionModel.keras')

model = load_model('emotionRecognitionModel.keras')

predictions = model.predict(trainImageList)
predictedClasses = np.argmax(predictions, axis=1)