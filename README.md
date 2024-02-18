# from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
IMAGE_SIZE = [224, 224]
train_path = "/content/drive/MyDrive/PDS/chest_xray/train"
test_path = "/content/drive/MyDrive/PDS/chest_xray/test"
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
for layer in vgg.layers:
layer.trainable = False
folders = glob("/content/drive/MyDrive/PDS/chest_xray/train")
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
# view the structure of the model
model.summary()
Model: "model_4"
_________________________________________________________________
 Layer (type) Output Shape Param # 
=================================================================
 input_5 (InputLayer) [(None, 224, 224, 3)] 0 
 
 block1_conv1 (Conv2D) (None, 224, 224, 64) 1792 
 block1_conv2 (Conv2D) (None, 224, 224, 64) 36928 
 block1_pool (MaxPooling2D) (None, 112, 112, 64) 0 
 block2_conv1 (Conv2D) (None, 112, 112, 128) 73856 
18/02/2024, 16:17 PDS.ipynb - Colaboratory
https://colab.research.google.com/drive/1FafxU8_LlKD2oTZC-eFMEUVKbvCz4ybR#printMode=true 2/4
 
 block2_conv2 (Conv2D) (None, 112, 112, 128) 147584 
 block2_pool (MaxPooling2D) (None, 56, 56, 128) 0 
 block3_conv1 (Conv2D) (None, 56, 56, 256) 295168 
 block3_conv2 (Conv2D) (None, 56, 56, 256) 590080 
 block3_conv3 (Conv2D) (None, 56, 56, 256) 590080 
 block3_pool (MaxPooling2D) (None, 28, 28, 256) 0 
 block4_conv1 (Conv2D) (None, 28, 28, 512) 1180160 
 block4_conv2 (Conv2D) (None, 28, 28, 512) 2359808 
 block4_conv3 (Conv2D) (None, 28, 28, 512) 2359808 
 block4_pool (MaxPooling2D) (None, 14, 14, 512) 0 
 block5_conv1 (Conv2D) (None, 14, 14, 512) 2359808 
 block5_conv2 (Conv2D) (None, 14, 14, 512) 2359808 
 block5_conv3 (Conv2D) (None, 14, 14, 512) 2359808 
 block5_pool (MaxPooling2D) (None, 7, 7, 512) 0 
 flatten_4 (Flatten) (None, 25088) 0 
 dense_4 (Dense) (None, 1) 25089 
=================================================================
Total params: 14739777 (56.23 MB)
Trainable params: 25089 (98.00 KB)
Non-trainable params: 14714688 (56.13 MB)
_________________________________________________________________
model.compile(
loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy']
)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
18/02/2024, 16:17 PDS.ipynb - Colaboratory
https://colab.research.google.com/drive/1FafxU8_LlKD2oTZC-eFMEUVKbvCz4ybR#printMode=true 3/4
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory("/content/drive/MyDrive/PDS/chest_xray/t
target_size = (224, 224),
batch_size = 10,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory("/content/drive/MyDrive/PDS/chest_xray/test",
target_size = (224, 224),
batch_size = 10,
class_mode = 'categorical')
Found 5216 images belonging to 2 classes.
Found 624 images belonging to 2 classes.
r = model.fit(
training_set,
validation_data=test_set,
epochs=1,
steps_per_epoch=len(training_set),
validation_steps=len(test_set)
)
522/522 [==============================] - 3396s 7s/step - loss: 0.7033 - accuracy: 0
import tensorflow as tf
from keras.models import load_model
model.save('chest_xray.keras')
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model=load_model('chest_xray.keras')
img=image.load_img("/content/drive/MyDrive/PDS/chest_xray/val/NORMAL/NORMAL2-IM-1431-0001
x=image.img_to_array(img)
18/02/2024, 16:17 PDS.ipynb - Colaboratory
https://colab.research.google.com/drive/1FafxU8_LlKD2oTZC-eFMEUVKbvCz4ybR#printMode=true 4/4
x=np.expand_dims(x, axis=0)
img_data=preprocess_input(x)
classes=model.predict(img_data)
1/1 [==============================] - 0s 472ms/step
result=int(classes[0][0])
if result==0:
print("Person is Affected By PNEUMONIA")
else:
print("Result is Normal")
Result is Norma
