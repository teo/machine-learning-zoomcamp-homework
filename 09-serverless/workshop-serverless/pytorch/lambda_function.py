import onnxruntime as ort
from keras_image_helper import create_preprocessor
import numpy as np

onnx_model_path = 'clothing_classifier_mobilenet_v2_latest.onnx'
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

def preprocess_pytorch_style(X):
    # X: shape (1, 299, 299, 3), dtype=float32, values in [0, 255]
    X = X / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

    # Convert NHWC → NCHW
    # from (batch, height, width, channels) → (batch, channels, height, width)
    X = X.transpose(0, 3, 1, 2)

    # Normalize
    X = (X - mean) / std

    return X.astype(np.float32)

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

preprocessor = create_preprocessor(preprocess_pytorch_style, target_size=(224, 224))

def lambda_handler( event, context):
    url = event['url']
    X = preprocessor.from_url(url)

    preds = session.run([output_name], {input_name: X})
    float_predictions = preds[0][0].tolist()

    result = dict(zip(classes, float_predictions))

    return result
