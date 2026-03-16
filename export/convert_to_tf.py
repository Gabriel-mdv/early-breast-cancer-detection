"""
Convert ONNX model to TensorFlow format.
"""

import onnx
import tf2onnx

def convert_onnx_to_tf(onnx_path: str, tf_path: str):
    onnx_model = onnx.load(onnx_path)
    tf2onnx.convert.from_onnx(onnx_model, output_path=tf_path)
