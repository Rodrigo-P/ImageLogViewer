from utils.imagelog_functions import apply_windowing

from matplotlib import pyplot as plt

import numpy as np


def colorido_function(loaded_data, start, end):
    image_array = loaded_data["equalized"][start:end, :]
    image_array = plt.cm.YlOrBr_r(image_array)[:, :, :3]

    return image_array


def bw_function(loaded_data, start, end, rgb=True):
    image_array = loaded_data["equalized"][start:end, :]
    if rgb:
        image_array = np.repeat(image_array[..., None], 3, axis=2)
    return image_array


def windowing_function(loaded_data, start, end, center, width, rgb=True):
    signal_array = loaded_data["signal"][start:end, :]
    image_array = apply_windowing(width, center, signal_array)
    if rgb:
        # image_array = np.repeat(image_array[..., None], 3, axis=2)
        image_array = plt.cm.YlOrBr_r(image_array)[:, :, :3]
    return image_array


def nodulo_function(loaded_data, start, end, rgb=True, black_pad=True):
    image = windowing_function(loaded_data, start, end, 50, 30, rgb)

    # if black_pad and "mask" in loaded_data:
    # image[loaded_data["mask"][start:end, :]] = 0

    return image


def vug_function(loaded_data, start, end, rgb=True):
    return windowing_function(loaded_data, start, end, 195, 50, rgb)


def composto_function(loaded_data, start, end):
    signal_array = loaded_data["signal"][start:end, :]
    image_array = np.zeros((signal_array.shape + (3,)), dtype=np.float32)

    image_array[:, :, 0] = loaded_data["equalized"][start:end, :]
    image_array[:, :, 1] = nodulo_function(loaded_data, start, end, rgb=False)
    image_array[:, :, 2] = vug_function(loaded_data, start, end, rgb=False)

    return image_array


def blackpad_function(loaded_data, start, end):
    image_array = colorido_function(loaded_data, start, end)
    if "mask" in loaded_data:
        image_array[loaded_data["mask"][start:end, :]] = 0

    return image_array


def make_windowing_function(center, width):
    return lambda loaded_data, start, end: windowing_function(
        loaded_data, start, end, center, width, True
    )


PROCESSING_FUNCTION = {
    "Colorido": colorido_function,
    "Nódulo": nodulo_function,
    "B&W": bw_function,
    "Cavidade": vug_function,
    "Black Pads": blackpad_function,
    "Composto": composto_function,
    "Custom Window": make_windowing_function(195, 50),
}
