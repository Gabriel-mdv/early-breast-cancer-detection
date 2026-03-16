"""
Quantize TensorFlow Lite model from FP32 to INT8.
"""

import tensorflow as tf

def quantize_tflite_model(tflite_path: str, quantized_path: str):
    converter = tf.lite.TFLiteConverter.from_saved_model(tflite_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    quantized_model = converter.convert()
    with open(quantized_path, 'wb') as f:
        f.write(quantized_model)
