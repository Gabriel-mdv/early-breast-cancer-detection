"""
Convert TensorFlow model to TensorFlow Lite format.
"""

import tensorflow as tf

def convert_tf_to_tflite(tf_model_path: str, tflite_path: str):
    model = tf.keras.models.load_model(tf_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
