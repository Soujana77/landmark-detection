# Category Classification Model

## Model
EfficientNetB0 Transfer Learning

## Classes
- Gothic
- Modern
- Mughal
- Neoclassical
- Pagodas
- Pyramids

## Architecture

Input Image (224x224)
↓
Data Augmentation
↓
EfficientNetB0
↓
Global Average Pooling
↓
Dropout
↓
Dense(6)
↓
Softmax Output

## Goal
Predict the category of a landmark image.