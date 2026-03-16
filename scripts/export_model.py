
"""
CLI script for exporting MobileFCMViTv3 model.
"""

import argparse
from export.export_onnx import export_to_onnx
from export.convert_to_tf import convert_onnx_to_tf
from export.convert_to_tflite import convert_tf_to_tflite
from export.quantization import quantize_tflite_model

def main():
    parser = argparse.ArgumentParser(description='Export MobileFCMViTv3 model')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[1, 1, 224, 224])
    parser.add_argument('--onnx_path', type=str, default='export/model.onnx')
    parser.add_argument('--tf_path', type=str, default='export/model_tf')
    parser.add_argument('--tflite_path', type=str, default='export/model.tflite')
    parser.add_argument('--quantized_path', type=str, default='export/model_quant.tflite')
    args = parser.parse_args()
    # ...model setup...
    # export_to_onnx(model, tuple(args.input_shape), args.onnx_path)
    # convert_onnx_to_tf(args.onnx_path, args.tf_path)
    # convert_tf_to_tflite(args.tf_path, args.tflite_path)
    # quantize_tflite_model(args.tflite_path, args.quantized_path)
    print('Model export started.')

if __name__ == '__main__':
    main()
