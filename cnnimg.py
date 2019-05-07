"""
dumpimages.py
    Downloads all the images on the supplied URL, and saves them to the
    specified output file ("/test/" by default)

Usage:
    python dumpimages.py http://example.com/ [output]
"""
from bs4 import BeautifulSoup as bs
from urllib.request import (
    urlopen, urlparse, urlunparse, urlretrieve)
import os
import sys

def main(url, out_folder="/test/"):
    """Downloads all the images at 'url' to /test/"""
    soup = bs(urlopen(url),features='html.parser')
    parsed = list(urlparse(url))

    for image in soup.findAll("img"):
        print("Image: %(src)s" % image)
        filename = image["src"].split("/")[-1]
        parsed[2] = image["src"]
        outpath = os.path.join(out_folder, filename)
        if image["src"].lower().startswith("http"):
            urlretrieve(image["src"], outpath)
        else:
            urlretrieve(urlunparse(parsed), outpath)

def _usage():
    print("usage: python dumpimages.py http://example.com [outpath]")

if __name__ == "__main__":
    url = sys.argv[-1]
    out_folder = "/test/"
    if not url.lower().startswith("http"):
        out_folder = sys.argv[-1]
        url = sys.argv[-2]
        if not url.lower().startswith("http"):
            _usage()
            sys.exit(-1)
    main(url, out_folder)
    
    
    
    
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from skimage.io import imread
from skimage.transform import resize
import numpy as np
 
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Fitting images to CNN

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('Dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('Dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
                                            
                                            
# Reading images from training set                                   
classifier.fit_generator(training_set,
                         steps_per_epoch = (8000/32),
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = (2000/32))
                         


                         
#Predicting images
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/maria/Development/Webfiltering/images/im2.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
 prediction = 'nude'
 print(prediction)
else:
 prediction = 'nonnude'
 print(prediction)
    
    
    
    
    
    

