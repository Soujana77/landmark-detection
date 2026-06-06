import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model(
    "../model/landmark_category_classifier.keras"
)

# Category names
class_names = [
    "Gothic",
    "Modern",
    "Mughal",
    "Neoclassical",
    "Pagodas",
    "Pyramids"
]

# Test image
img_path = "test.jpg"

img = image.load_img(img_path, target_size=(224, 224))

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("\nPrediction:", predicted_class)
print("Confidence:", round(float(confidence) * 100, 2), "%")