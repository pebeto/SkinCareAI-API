import io

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

from util.constants import RGB

def map_format(image):
    return Image.open(io.BytesIO(image))

def prepare_image(image, target):
    if image.mode != RGB:
        image = image.convert(RGB)
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image
