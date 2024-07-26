from models.segmentation.SLIC_DBSCAN.transform import (
    SegmentImageLogTale as SLICDBSCANSegmentation,
)

from utils.imagelog_functions import apply_windowing

from multiprocessing import Process

import numpy as np
import radiomics
import logging
import torch
import os


TRANSFORM_ENUMS = {
    "SLIC": SLICDBSCANSegmentation,
}


def make_segmentation_transform(selected_type, package_path):
    # Create the segmentation
    return TRANSFORM_ENUMS[selected_type](package_path)


class AsyncSegmentation(Process):
    def __init__(
        self,
        segmentation_mode: str,
        resistive_data: dict,
        package_path: str,
        imglog_name: str,
    ):
        super().__init__()
        self.cache_folder = f"{package_path}/cache/{imglog_name}/{segmentation_mode}"

        self.segmentation_transform = make_segmentation_transform(
            segmentation_mode, package_path
        )
        self.windowed_data = apply_windowing(50, 195, resistive_data["signal"])
        self.running = True

        self.set_logging()

    def set_logging(self):
        log_file = f"{self.cache_folder}/radiomics.log"
        handler = logging.FileHandler(filename=log_file, mode="w")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        radiomics.logger.addHandler(handler)
        # Set logging verbosity
        logging.getLogger("radiomics.glcm").setLevel(logging.ERROR)

    def run(self):
        data_width = self.windowed_data.shape[1]
        num_tales = self.windowed_data.shape[0] // data_width

        os.makedirs(self.cache_folder, exist_ok=True)

        for tale_number in range(num_tales):

            cache_file_path = f"{self.cache_folder}/{tale_number}.npy"

            if os.path.exists(cache_file_path):
                continue

            image_array = self.windowed_data[
                data_width * tale_number : data_width * (tale_number + 1), :
            ]

            # Apply the segmentation
            tensor_input = torch.Tensor(np.expand_dims(image_array, axis=0))
            slic_superpixels = self.segmentation_transform(tensor_input)
            label_array = np.array(slic_superpixels).squeeze()

            sorted_label_array = np.zeros(label_array.shape, label_array.dtype)
            unique, counts = np.unique(label_array, return_counts=True)
            label_descending_order = [
                unique[index] for index in np.flip(np.argsort(counts))
            ]

            for index, label in enumerate(label_descending_order):
                sorted_label_array[label_array == label] = index

            np.save(
                file=cache_file_path,
                arr=sorted_label_array,
            )

    def stop(self):
        self.running = False
