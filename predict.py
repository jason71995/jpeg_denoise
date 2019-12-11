from model.resnet import build_model
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", default=None, type=str, help="Image path.")
parser.add_argument("-m", "--model", default=None, type=str, help="Model path.")
args = parser.parse_args()

model = build_model(filters=64,block=16)
model.load_weights(args.model)

input_image = Image.open(args.image).convert("RGB")
input_image = np.expand_dims(np.asarray(input_image, "float32") / 127.5 - 1, axis=0)

pred_image = model.predict(input_image)
pred_image = Image.fromarray(np.clip((pred_image[0] + 1) * 127.5, 0, 255).astype("uint8"), "RGB")
pred_image.save("img_jpeg_denoise.png")