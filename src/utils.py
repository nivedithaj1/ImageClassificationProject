import os
import numpy as np
from PIL import Image, ImageOps


def get_groceries():
    groceries = [item for item in os.listdir(
        'models') if os.path.isdir(os.path.join('models', item))]
    return groceries


def load_image(img, inp_shape):
    with Image.open(img) as _img:
        image = _img.resize(inp_shape)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
    return image
