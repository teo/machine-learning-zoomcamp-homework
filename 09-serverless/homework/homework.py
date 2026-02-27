#!/usr/bin/env python3

from onnxruntime import InferenceSession
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

model_path = 'hair_classifier_v1.onnx'
session = InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# print input and output name
print(f"Input Name: {input_name}")
print(f"Output Name: {output_name}")

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

target_size=(200,200)

image=download_image("https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg")
image=prepare_image(image, target_size)


# Set up transforms
from torchvision import transforms

val_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])

# apply val_transforms to input image
image = val_transforms(image)
print(image.shape) # should print torch.Size([3, 200, 200])
print(image[0][0][0])

# apply the inference session to the image
output = session.run(input_feed={"input": image.unsqueeze(0).numpy()}, output_names=["output"])

print(output) # should print a numpy array with shape (1, 2)
