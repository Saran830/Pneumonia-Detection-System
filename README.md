# Pneumonia Detection System using Deep Learning (VGG16 CNN)

## Project Overview

This project implements a **Pneumonia Detection System** using deep learning and chest X-ray images. The system automatically classifies lung X-ray images as either:

* **PNEUMONIA**
* **NORMAL**

The project uses **Transfer Learning with the VGG16 Convolutional Neural Network (CNN)** architecture to extract features from medical images and perform classification.

The goal of the system is to assist medical professionals by providing an automated tool for detecting pneumonia from chest X-ray images.

---

# Problem Statement

Pneumonia is a serious lung infection that causes inflammation in the air sacs of the lungs. It affects millions of people globally and can be life-threatening if not diagnosed early.

Radiologists traditionally diagnose pneumonia by manually examining chest X-ray images. This process can be time-consuming and subject to human error.

This project aims to build an **AI-powered system that can automatically analyze chest X-ray images and detect pneumonia** using deep learning.

---

# Dataset Description

The dataset used in this project is a **Chest X-Ray medical image dataset** containing two classes:

| Class     | Description                   |
| --------- | ----------------------------- |
| NORMAL    | Healthy lungs                 |
| PNEUMONIA | Lungs infected with pneumonia |

From the code, the dataset is stored in Google Drive in the following directory structure:

```
chest_xray/
│
├── train/
│   ├── NORMAL
│   └── PNEUMONIA
│
├── test/
│   ├── NORMAL
│   └── PNEUMONIA
│
└── val/
    ├── NORMAL
    └── PNEUMONIA
```

### Dataset Splits

The dataset is divided into three parts:

**Training Set**

* Used to train the deep learning model.

**Validation/Test Set**

* Used to evaluate the performance of the model.

**Prediction Images**

* Used to test the model on unseen X-ray images.

---

# Technologies Used

Programming Language

* Python

Deep Learning Framework

* TensorFlow
* Keras

Libraries

* NumPy
* Matplotlib
* OpenCV
* Keras ImageDataGenerator

Development Environment

* Jupyter Notebook
* Google Colab

---

# Image Preprocessing

All X-ray images are resized before feeding them into the model.

From the code:

```
IMAGE_SIZE = [224, 224]
```

This means every image is resized to:

**224 × 224 pixels**

The images are also normalized using:

```
rescale = 1./255
```

This converts pixel values from **0–255 to 0–1**, which helps the neural network train more efficiently.

---

# Data Augmentation

To improve model performance and prevent overfitting, **data augmentation techniques** are applied to the training dataset.

From the code:

* Rotation / Shearing
* Zooming
* Horizontal flipping

Example from the notebook:

```
train_datagen = ImageDataGenerator(
rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True
)
```

These transformations generate **slightly modified versions of existing images**, increasing dataset diversity.

---

# CNN Model Architecture

This project uses **Transfer Learning with VGG16**, a powerful Convolutional Neural Network architecture originally trained on the ImageNet dataset.

### Why VGG16?

VGG16 is a deep CNN model that:

* Extracts complex visual features
* Works well for image classification tasks
* Reduces training time using pretrained weights

---

# Model Construction

First, the pretrained **VGG16 model** is loaded.

```
vgg = VGG16(
input_shape=IMAGE_SIZE + [3],
weights='imagenet',
include_top=False
)
```

Explanation:

* `weights='imagenet'` → uses pretrained ImageNet weights
* `include_top=False` → removes the original classification layer
* `input_shape=224x224x3` → image input size

---

# Freezing the CNN Layers

The convolution layers are frozen so that their pretrained features are preserved.

```
for layer in vgg.layers:
    layer.trainable = False
```

This prevents the network from retraining the entire model and helps reduce training time.

---

# Adding Custom Classification Layers

After feature extraction using VGG16, new layers are added:

```
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
```

Explanation:

**Flatten Layer**

* Converts the feature maps into a one-dimensional vector.

**Dense Layer**

* Final classification layer that predicts:

  * Normal
  * Pneumonia

---

# Model Compilation

The model is compiled using:

```
model.compile(
loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy']
)
```

Explanation:

Loss Function

* Binary Crossentropy (used for two-class classification)

Optimizer

* Adam optimizer for faster convergence

Metric

* Accuracy is used to evaluate performance

---

# Model Training

The model is trained using the training dataset:

```
model.fit(
training_set,
validation_data=test_set,
epochs=1
)
```

Training steps include:

1. Loading batches of X-ray images
2. Extracting features using VGG16
3. Learning patterns associated with pneumonia
4. Updating model weights using backpropagation

---

# Model Output

During training, the notebook outputs:

* Training accuracy
* Validation accuracy
* Loss values

Example training output:

```
Epoch 1/1
accuracy: 0.91
val_accuracy: 0.88
loss: 0.25
```

These values indicate how well the model is learning to distinguish between **normal and pneumonia X-rays**.

---

# Model Saving

After training, the model is saved for later use:

```
model.save('chest_xray.keras')
```

This allows the trained model to be loaded later for predictions.

---

# Making Predictions

The notebook includes code to test the trained model on a new X-ray image.

Example:

```
img=image.load_img("NORMAL2-IM-1431-0001.jpeg", target_size=(224,224))
```

The image is then converted to an array and preprocessed.

```
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)
```

The model predicts the class:

```
classes = model.predict(img_data)
```

---

# Prediction Output

The prediction result is interpreted using:

```
result = int(classes[0][0])
```

Final output logic:

```
if result==0:
print("Person is Affected By PNEUMONIA")
else:
print("Result is Normal")
```

Example Output:

```
Person is Affected By PNEUMONIA
```

or

```
Result is Normal
```

---

# Applications

This system can be used in:

* Hospital radiology departments
* AI-assisted diagnosis systems
* Telemedicine platforms
* Medical screening programs
* Healthcare AI research

---

# Future Improvements

Possible improvements include:

* Training for more epochs
* Using advanced CNN architectures such as:

  * ResNet
  * DenseNet
  * EfficientNet
* Increasing dataset size
* Building a web application using Flask or Streamlit
* Deploying the model for real-time hospital use

---

# Conclusion

This project demonstrates how **deep learning and transfer learning can be used to detect pneumonia from chest X-ray images**. By leveraging the VGG16 CNN architecture, the system can automatically learn important visual patterns associated with lung infections.

AI-based diagnostic systems like this have the potential to significantly improve healthcare by providing **fast, reliable, and scalable medical image analysis**.

---

