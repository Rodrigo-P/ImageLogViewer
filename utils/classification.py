from models.classification.VGG16.transform import (
    build_classifier as VGG16classification,
)

import numpy as np
import torch
import json
import os

CLASSIFIER_ENUMS = {
    "VGG16": VGG16classification,
}


class ClassificationModule:
    def __init__(
        self,
        classification_mode: str,
        resistive_data: dict,
        PACKAGEPATH: str,
        imglog_name: str,
    ):
        self.classification_transform = CLASSIFIER_ENUMS[classification_mode](
            PACKAGEPATH
        )
        self.tale_data = resistive_data["signal"]

        self.cache_folder = f"{PACKAGEPATH}/cache/{imglog_name}/{classification_mode}"
        os.makedirs(self.cache_folder, exist_ok=True)

        json_path = (
            f"{PACKAGEPATH}/models/classification/{classification_mode}/class_dict.json"
        )
        with open(json_path, "r", encoding="utf-8") as json_file:
            self.class_dict = json.load(json_file)

    def run(self):
        cache_file_path = f"{self.cache_folder}/class_prediction.json"
        # if os.path.exists(cache_file_path):
        #     return

        data_width = self.tale_data.shape[1]
        num_tales = self.tale_data.shape[0] // data_width

        prediction_dictionary = {}

        for tale_number in range(num_tales):
            image_array = self.tale_data[
                data_width * tale_number : data_width * (tale_number + 1), :
            ]

            # Apply the classification
            tensor_input = torch.Tensor(
                np.repeat(image_array[np.newaxis, ...], 3, axis=0)
            ).unsqueeze(0)
            tensor_input = (tensor_input - tensor_input.min()) / (
                tensor_input.max() - tensor_input.min()
            )
            # tensor_input -= torch.min(tensor_input)
            # tensor_input /= torch.max(tensor_input)
            # 224, 224
            out = self.classification_transform(tensor_input)
            out = torch.argmax(out)

            prediction_dictionary[tale_number] = self.class_dict[str(int(out))]

        with open(cache_file_path, "w", encoding="utf-8") as json_file:
            json.dump(prediction_dictionary, json_file, indent=4)
