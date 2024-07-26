from utils import normalization

from PIL import Image

import numpy as np
import pandas
import cv2

INT_PIXEL_CONVERSION = (2**16) - 1
INT_PIXEL_TYPE = np.uint16


def matrix_2D_gray_1channel(matrix: np.ndarray):
    matrix = normalization.image_from_data(matrix)
    return matrix


def load_image_log_csv(csv_path: str):
    # Compute mask and crop useful subimage.
    INVALID_VALUE = -9999.0  # Constant value indicading invalid value.
    MAXIMUM_VALUE = 40000  # Constant value indicading maximum value.

    # Load raw acoustic image log table.
    table = pandas.read_csv(
        csv_path, sep=";", decimal=",", dtype=np.float32
    ).to_numpy()  # Lê o arquivo csv e converte para um array numpy

    # Keep signal, depth, and angles.
    depths = table[:, 0]  # Extrair a amplitude do poço.
    signal = (
        table[:, 1:-1] if np.all(np.isnan(table[:, -1])) else table[:, 1:]
    )  # Obter o sinal ignorando a primeira coluna (amplitude do poço) e a última coluna se todos os valores na última coluna são NaN.
    (
        signal_height,
        signal_width,
    ) = signal.shape  # Resolução do sinal e da máscara que o acompanha.
    angles = np.linspace(
        0.0, 2 * np.pi, signal_width, endpoint=False, dtype=np.float32
    )  # Calcular ângulos associados à colunas do sinal.

    mask = np.logical_or(
        np.isnan(signal), np.isinf(signal)
    )  # Máscara booleana que indica com True quais pixes do sinal são inválidos e com False quais são válidos.
    signal[mask] = INVALID_VALUE
    signal[signal < INVALID_VALUE] = INVALID_VALUE
    mask = signal == INVALID_VALUE

    valid_row = np.any(
        np.logical_not(mask), axis=1
    )  # Vetor booleano com True se pelo menos um dos valores da linha (eixo) for válido.
    end = signal_height - np.argmax(
        valid_row[::-1]
    )  # Índice da linha seguinte à última linha com algum valor válido.
    begin = np.argmax(valid_row)  # Índice da primeira linha com algum valor válido.

    depths = depths[begin:end]  # Profundidades sem a parte inválida inicial e final.
    signal = signal[begin:end, ...]  # Sinal sem a parte inválida inicial e final.
    mask = mask[begin:end, ...]  # Máscara sem a parte inválida inicial e final.

    # Move invalid values.
    signal[signal == INVALID_VALUE] = np.unique(signal)[1] - 10
    # Erase big outiliers.
    # signal[signal > MAXIMUM_VALUE] = MAXIMUM_VALUE

    loaded_data = {"signal": signal, "mask": mask, "depth": depths}

    # Return the image log.
    return loaded_data


def align_acoustic_signal(acoustic, resistive):
    acoustic_start = np.min(acoustic["depth"])
    resist_start = np.min(resistive["depth"])
    acoustic_end = np.max(acoustic["depth"])
    resist_end = np.max(resistive["depth"])

    bottom_pad_size = 0
    top_pad_size = 0

    # Top align
    if acoustic_start < resist_start:
        for i in range(acoustic["depth"].shape[0]):
            if acoustic["depth"][i] > resistive["depth"][0]:
                first_pos = i
                break
        acoustic["signal"] = acoustic["signal"][first_pos:, :]
        acoustic["depth"] = acoustic["depth"][first_pos:]
        acoustic["mask"] = acoustic["mask"][first_pos:, :]
    elif acoustic_start > resist_start:
        for i in range(acoustic["depth"].shape[0]):
            if resistive["depth"][i] > acoustic["depth"][0]:
                top_pad_size = i
                break

    # Bottom align
    if acoustic_end > resist_end:
        for i in range(1, acoustic["depth"].shape[0]):
            if acoustic["depth"][-1 * i] < resistive["depth"][-1]:
                last_pos = acoustic["depth"].shape[0] - i
                break
        acoustic["signal"] = acoustic["signal"][:last_pos, :]
        acoustic["depth"] = acoustic["depth"][:last_pos]
        acoustic["mask"] = acoustic["mask"][:last_pos, :]
    elif acoustic_end < resist_end:
        for i in range(1, acoustic["depth"].shape[0]):
            if resistive["depth"][-1 * i] < acoustic["depth"][-1]:
                bottom_pad_size = i
                break

    # Resize to shape minus padding
    resize_shape = (
        resistive["signal"].shape[1],
        resistive["signal"].shape[0] - (top_pad_size + bottom_pad_size),
    )
    acoustic["signal"] = cv2.resize(acoustic["signal"], resize_shape)

    # Add padding to acoustic data
    if bottom_pad_size != 0:
        padding = np.zeros(shape=(bottom_pad_size, resistive["signal"].shape[1]))
        acoustic["signal"] = np.concatenate((padding, acoustic["signal"]), axis=0)

    if top_pad_size != 0:
        padding = np.zeros(shape=(top_pad_size, resistive["signal"].shape[1]))
        acoustic["signal"] = np.concatenate((acoustic["signal"], padding), axis=0)


def get_label_data(file_path: str):
    target_data = {"boxes": [], "labels": []}

    text_file = open(file_path)

    for line in text_file:
        line = line.rstrip()
        value_list = line.split()
        class_index = value_list.pop(0)

        target_data["labels"].append(int(class_index))
        target_data["boxes"].append([float(el) for el in value_list])

    target_data["labels"] = np.array(target_data["labels"])
    target_data["boxes"] = np.array(target_data["boxes"])

    return target_data


def draw_rectangle_overlay(image, x0, x1, y0, y1, color, alpha=0.2):
    image[y0:y1, x0:x1, :] *= 1 - alpha
    image[y0:y1, x0:x1, :] += [i * alpha for i in color]


def image_from_signal(signal_data):
    image_array = matrix_2D_gray_1channel(signal_data)
    image_array = image_array.astype(np.float32) / INT_PIXEL_CONVERSION
    return image_array


def normalize_array(input_array):
    input_array -= np.min(input_array)
    input_array /= np.max(input_array)
    return input_array


def float32_to_uint8(input_array):
    return (input_array * 255).astype(np.uint8)


def float32_to_pil(input_array: np.ndarray):
    image_array = np.copy(input_array)
    if len(image_array.shape) != 3:
        image_array = np.repeat(image_array[..., None], 3, axis=2)
        # image_array = plt.cm.YlOrBr(image_array)[:, :, :3]

    image_array = float32_to_uint8(image_array)

    return Image.fromarray(image_array)


def apply_windowing(width: int, center: int, img_data: np.ndarray):
    result_data = np.copy(img_data)
    c = center - 0.5
    w = width - 1.0

    min_mask = result_data <= (c - 0.5 * w)
    max_mask = result_data > (c + 0.5 * w)

    result_data -= c
    result_data /= w
    result_data += 0.5

    result_data[min_mask] = 0
    result_data[max_mask] = 1

    return 1 - result_data


def draw_rectangle_border(image, x0, x1, y0, y1, color, width=4):
    y1 = min(y1, image.shape[0])
    x1 = min(x1, image.shape[1])
    y0 = max(y0, width)
    x0 = max(x0, width)

    image[y0 - width : y0, x0 - width : x1 + width, :] = color
    image[y1 : y1 + width, x0 - width : x1 + width, :] = color
    image[y0 - width : y1 + width, x0 - width : x0, :] = color
    image[y0 - width : y1 + width, x1 : x1 + width, :] = color
