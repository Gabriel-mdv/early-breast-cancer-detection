"""
Custom training, evaluation, and export workflows for research.
"""

from training.trainer import Trainer
from evaluation.evaluation_pipeline import EvaluationPipeline
from export.export_onnx import export_to_onnx
from export.convert_to_tf import convert_onnx_to_tf
from export.convert_to_tflite import convert_tf_to_tflite
from export.quantization import quantize_tflite_model

class CustomTrainingWorkflow:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
    def run(self):
        # Custom schedule, multi-task, federated, etc.
        self.trainer.train()

class CustomEvaluationWorkflow:
    def __init__(self, pipeline: EvaluationPipeline):
        self.pipeline = pipeline
    def run(self):
        # Custom metrics, hooks, etc.
        return self.pipeline.evaluate()

class CustomExportWorkflow:
    def __init__(self, model, input_shape, paths):
        self.model = model
        self.input_shape = input_shape
        self.paths = paths
    def run(self):
        export_to_onnx(self.model, self.input_shape, self.paths['onnx'])
        convert_onnx_to_tf(self.paths['onnx'], self.paths['tf'])
        convert_tf_to_tflite(self.paths['tf'], self.paths['tflite'])
        quantize_tflite_model(self.paths['tflite'], self.paths['quantized'])
