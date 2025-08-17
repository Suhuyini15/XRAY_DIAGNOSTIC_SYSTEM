import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("converted_model")

# Class labels from your training
CLASS_NAMES = ["uninfected", "pneumonia", "pneumothorax", "mass", "effusion", "copd"]

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_disease(image_path):
    img_array = preprocess_image(image_path)

    try:
        # Normal Keras model
        preds = model.predict(img_array)[0]
    except AttributeError:
        # Handle SavedModel (_UserObject)
        infer = model.signatures["serving_default"]
        outputs = infer(tf.constant(img_array))
        
        # Get the first tensor in the output dict
        preds = list(outputs.values())[0].numpy()[0]

    predictions = {CLASS_NAMES[i]: float(preds[i] * 100) for i in range(len(CLASS_NAMES))}
    sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

    top_class = max(predictions, key=predictions.get)
    top_confidence = predictions[top_class]

    return top_class, top_confidence, sorted_predictions
