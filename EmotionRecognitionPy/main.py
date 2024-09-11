from PIL import Image
import pandas as pdb
import numpy as np

dataset = pdb.read_csv('train.csv')

emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'suprise', 6: 'neutral'}

imageDictionary = {emotion: [] for emotion in emotions.values()}

for index, row in dataset.iterrows():
    # row[] retrieves the value from current row specified by heading
    emotion = emotions[row['emotion']]
    pixels = row['pixels']
    image = np.fromstring(pixels, sep=' ', dtype=float).reshape(48,48)
    
    imageDictionary[emotion].append(image)
    
for emotion in imageDictionary:
    imageDictionary[emotion] = np.array(imageDictionary[emotion])


    
print(imageDictionary['angry'].shape)