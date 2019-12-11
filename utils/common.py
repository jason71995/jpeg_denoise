import os
import glob
import numpy as np
import random
from PIL import Image, ImageEnhance
from io import BytesIO

def get_jpg_image(image,quality):
    output = BytesIO()
    image.save(output,format="JPEG", quality=quality)
    return Image.open(output).convert("RGB")

def data_loader(folder, batch_size, patch_size, augmentation, preload_all_image):
    image_paths = glob.glob(os.path.join(folder, "*.png"))

    images = []
    if preload_all_image:
        for path in image_paths:
            images.append(Image.open(path).convert("RGB"))

    while True:
        rand_idx = np.random.permutation(len(image_paths))[:batch_size]

        batch_raw_images = []
        batch_jpg_images = []

        for i in rand_idx:
            if preload_all_image:
                raw_image = images[i]
            else:
                raw_image = Image.open(image_paths[i]).convert("RGB")

            crop_left = random.randint(0, raw_image.size[0] - patch_size - 1)
            crop_top = random.randint(0, raw_image.size[1] - patch_size - 1)
            raw_image = raw_image.crop((crop_left, crop_top, crop_left + patch_size, crop_top + patch_size))

            if augmentation:
                raw_image = raw_image.transpose(Image.FLIP_TOP_BOTTOM) if np.random.randint(0, 2) is 0 else raw_image
                rand_rot = np.random.randint(1, 4)
                raw_image = raw_image.transpose(rand_rot) if rand_rot != 1 else raw_image
                raw_image = ImageEnhance.Color(raw_image).enhance(random.uniform(0.0, 2.0))
                raw_image = ImageEnhance.Contrast(raw_image).enhance(random.uniform(0.75, 1.25))
                raw_image = ImageEnhance.Brightness(raw_image).enhance(random.uniform(0.75, 1.25))

            jpg_image = get_jpg_image(raw_image, random.randint(1, 100))

            raw_image = np.asarray(raw_image, 'float32') / 127.5 - 1
            jpg_image = np.asarray(jpg_image, 'float32') / 127.5 - 1

            batch_jpg_images.append(jpg_image)
            batch_raw_images.append(raw_image)

        batch_jpg_images = np.asarray(batch_jpg_images)
        batch_raw_images = np.asarray(batch_raw_images)
        yield batch_jpg_images, batch_raw_images