import onnxruntime as ort
from keras_image_helper import create_preprocessor


onnx_model_path = 'clothing-model-new.onnx'
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name


preprocessor = create_preprocessor('xception', target_size=(299, 299))

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


def lambda_handler( event, context):
    url = event['url']
    X = preprocessor.from_url(url)

    preds = session.run([output_name], {input_name: X})
    float_predictions = preds[0][0].tolist()

    result = dict(zip(classes, float_predictions))

    return result
