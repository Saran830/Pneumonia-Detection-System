Pneumonia Detection System using Deep Learning (VGG16 Transfer Learning)

Project Overview

This project implements a Pneumonia Detection System using Transfer Learning with VGG16 in TensorFlow/Keras.

The model classifies chest X-ray images into two categories:

- Normal
- Pneumonia

A pre-trained VGG16 model (ImageNet weights) is used as a feature extractor, and a custom classification layer is added for binary classification.


Model Architecture

- Base Model: VGG16 (Pre-trained on ImageNet)
- include_top=False
- Frozen convolutional layers
- Custom Layers:
  - Flatten()
  - Dense() output layer

Model Parameters

- Total Parameters: 14,739,777
- Trainable Parameters: 25,089
- Non-Trainable Parameters: 14,714,688

This approach leverages powerful pre-trained visual features while keeping training lightweight.


Dataset Structure

The dataset follows the directory structure required by `flow_from_directory():

chest_xray/
│
├── train/
│ ├── NORMAL/
│ ├── PNEUMONIA/
│
├── test/
│ ├── NORMAL/
│ ├── PNEUMONIA/
│
└── val/
├── NORMAL/
├── PNEUMONIA/


- Training Images: 5216  
- Test Images: 624  
- Image Size: 224 × 224  


Technologies Used

- Python
- TensorFlow
- Keras
- VGG16 (Transfer Learning)
- Google Colab
- NumPy
- Matplotlib


Data Preprocessing

Training Data Augmentation

- Rescaling (1./255)
- Shear transformation
- Zoom transformation
- Horizontal flip

test Data

- Rescaling only

Data augmentation helps reduce overfitting and improve generalization.

Model Implementation

Load Pre-trained VGG16

python vgg = VGG16(input_shape=[224,224,3], weights='imagenet', include_top=False)

Initial Training Result:
Training Accuracy: 50%
Validation Accuracy: 50%

Output Interpretation:
0 → Pneumonia
1 → Normal

Training output:

522/522 [==============================]
loss: 0.7033  
accuracy: 0.5000  
val_loss: 0.6980  

Predicted output: Result is Normal
val_accuracy: 0.5000
