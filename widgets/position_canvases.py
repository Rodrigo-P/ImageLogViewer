# from INTERFACE.Visualizer.utils.display_functions import colorido_function
from utils.display_functions import colorido_function

from utils.imagelog_functions import (
    draw_rectangle_overlay,
    draw_rectangle_border,
    float32_to_pil,
)

from utils.style_formatting import STRUCTURE_WIDTH

from customtkinter import (
    CTkCanvas,
    CTkButton,
)

from typing import Dict, List
from PIL import Image, ImageTk


import customtkinter
import numpy as np
import math


class RulerCanvas(CTkCanvas):
    def __init__(self, root_class, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bind("<Configure>", self._resizer)

        self.root_class = root_class
        self.tale_size = 360

    def _draw(self):
        current_position: int = self.root_class.current_position
        resistive_data: dict = self.root_class.resistive_data
        number_of_rows: int = self.root_class.number_of_rows

        if resistive_data is None:
            return

        self.tale_size = resistive_data["equalized"].shape[1]
        height = self.winfo_height()

        ratio = (self.tale_size * number_of_rows) / height
        offset = current_position * self.tale_size
        self.ruler_texts = []
        start_pos = 10
        step = 10

        # start at `step` to skip line for `0`
        for y in range(start_pos, height, step):
            if y % 40 == start_pos:
                # draw longer line with text
                self.create_line(0, y, 10, y, width=2, fill="white")

                position = offset + int(ratio * y)
                text = f"{resistive_data['depth'][position]:.2f}"

                self.ruler_texts.append(
                    self.create_text(
                        12, y, text=text, fill="white", justify="left", anchor="w"
                    )
                )
            else:
                self.create_line(1, y, 4, y, fill="white")

        self.ruler_line_id = self.create_line(0, 0, 50, 0, fill="red", width=2)

    def _resizer(self, e):
        self.delete("all")
        self._draw()

    def update_values(self, number_of_rows, current_position, depth_values):
        height = self.winfo_height()

        ratio = (self.tale_size * number_of_rows) / height
        offset = current_position * self.tale_size
        step = 50

        # start at `step` to skip line for `0`
        for index, text_id in enumerate(self.ruler_texts):
            position = offset + int(ratio * (step * (index + 1)))

            self.itemconfigure(text_id, text=f"{depth_values[position]:.2f}")


class MiniatureLogCanvas(CTkCanvas):
    def __init__(self, root_class, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root_class = root_class
        self.miniature_image_id = self.create_image(
            0, 0, anchor="nw", image=self.root_class.blank_image_tk
        )
        self.tale_size = 360

        self.bind("<Configure>", self._resizer)
        self.bind("<Button-1>", self._click)

    def _click(self, event):
        previous_button: CTkButton = self.root_class.previous_button
        ruler_canvas: RulerCanvas = self.root_class.ruler_canvas
        next_button: CTkButton = self.root_class.next_button

        current_position: int = self.root_class.current_position
        resistive_data: dict = self.root_class.resistive_data
        number_of_rows: int = self.root_class.number_of_rows
        max_position: int = self.root_class.max_position

        imlog_height = resistive_data["signal"].shape[0]
        miniature_height = self.winfo_height()

        height_ratio = imlog_height / miniature_height
        y_value = event.y * height_ratio
        y_value //= self.tale_size

        current_position = min(max_position, int(y_value))

        if current_position == max_position:
            next_button.configure(state=customtkinter.DISABLED)
        else:
            next_button.configure(state=customtkinter.NORMAL)

        if current_position == 0:
            previous_button.configure(state=customtkinter.DISABLED)
        else:
            previous_button.configure(state=customtkinter.NORMAL)

        self.root_class.current_position = current_position

        ruler_canvas.update_values(
            depth_values=resistive_data["depth"],
            current_position=current_position,
            number_of_rows=number_of_rows,
        )
        self.root_class._update_image_grid()
        self.update_image()

    def _resizer(self, e):
        resistive_data: dict = self.root_class.resistive_data

        if not resistive_data:
            return

        height = e.height
        width = e.width

        self.miniature_image = float32_to_pil(resistive_data["equalized"]).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        self.miniature_image = colorido_function(
            {"equalized": np.array(self.miniature_image)[:, :, 0]}, 0, -1
        )
        self.update_image()

    def update_image(self):
        current_position: int = self.root_class.current_position
        resistive_data: dict = self.root_class.resistive_data
        number_of_rows: int = self.root_class.number_of_rows

        self.tale_size = resistive_data["equalized"].shape[1]

        image = np.copy(self.miniature_image)
        height = self.winfo_height()
        width = self.winfo_width()

        y0 = current_position * self.tale_size
        y1 = (current_position + number_of_rows) * self.tale_size
        y_ratio = height / resistive_data["equalized"].shape[0]

        y0 = math.floor(y0 * y_ratio)
        y1 = math.ceil(y1 * y_ratio)

        rectangle_color = [0, 0, 0.6]
        border_width = 2
        draw_rectangle_border(
            image,
            border_width,
            width - border_width,
            max(y0, border_width),
            min(y1, height - border_width),
            rectangle_color,
            border_width,
        )
        draw_rectangle_overlay(image, 0, width, y0, y1, rectangle_color)

        image = float32_to_pil(image)

        imagetk = ImageTk.PhotoImage(image)
        self.itemconfigure(self.miniature_image_id, image=imagetk)
        self.imgref = imagetk


class StructureViewCanvas(CTkCanvas):
    def __init__(
        self, root_class, bbox_data, bbox_label_color_dictionary, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.root_class = root_class

        self.color_dictionary: Dict = bbox_label_color_dictionary
        self.bbox_data: List[Dict] = bbox_data
        self.image_id = None

        self.column_dictionary = {}
        for index, label in enumerate(self.color_dictionary.keys()):
            self.column_dictionary[label] = index
        self.number_of_columns = len(self.color_dictionary)

        self.bind("<Configure>", self._resizer)
        self.bind("<Button-1>", self._click)

    def _click(self, event):
        miniature_canvas: RulerCanvas = self.root_class.miniature_canvas
        previous_button: CTkButton = self.root_class.previous_button
        ruler_canvas: RulerCanvas = self.root_class.ruler_canvas
        next_button: CTkButton = self.root_class.next_button

        current_position: int = self.root_class.current_position
        resistive_data: dict = self.root_class.resistive_data
        number_of_rows: int = self.root_class.number_of_rows
        max_position: int = self.root_class.max_position

        imlog_height = resistive_data["signal"].shape[0]
        miniature_height = self.winfo_height()

        height_ratio = imlog_height / miniature_height
        y_value = event.y * height_ratio
        y_value //= self.tale_size

        current_position = min(max_position, int(y_value))

        if current_position == max_position:
            next_button.configure(state=customtkinter.DISABLED)
        else:
            next_button.configure(state=customtkinter.NORMAL)

        if current_position == 0:
            previous_button.configure(state=customtkinter.DISABLED)
        else:
            previous_button.configure(state=customtkinter.NORMAL)

        self.root_class.current_position = current_position

        ruler_canvas.update_values(
            depth_values=resistive_data["depth"],
            current_position=current_position,
            number_of_rows=number_of_rows,
        )
        self.root_class._update_image_grid()
        miniature_canvas.update_image()

    def _resizer(self, e):
        self.update()
        self.make_image()

    def make_image(self):
        resistive_data: dict = self.root_class.resistive_data

        self.tale_size = resistive_data["equalized"].shape[1]

        width = self.number_of_columns * STRUCTURE_WIDTH
        height = self.winfo_height()
        self.configure(width=width)

        y_ratio = height / resistive_data["equalized"].shape[0]
        image = np.zeros((height, width, 3), dtype=np.float32)

        for i in range(self.number_of_columns):
            image[:, (STRUCTURE_WIDTH * i), :] = 1
            image[:, (STRUCTURE_WIDTH * i) - 1, :] = 1

        for position, bbox_dict in enumerate(resistive_data["bbox_data"]):
            y0 = math.floor(position * self.tale_size * y_ratio)
            y1 = math.ceil((position + 1) * self.tale_size * y_ratio)

            for label in bbox_dict["labels"]:
                column = self.column_dictionary[label]
                draw_rectangle_overlay(
                    image,
                    (column * STRUCTURE_WIDTH) + 1,
                    ((column + 1) * STRUCTURE_WIDTH) - 1,
                    y0,
                    y1,
                    self.color_dictionary[label]["rgb_float"],
                    alpha=1,
                )

        image = float32_to_pil(image)
        imagetk = ImageTk.PhotoImage(image)
        if self.image_id != None:
            self.itemconfigure(self.image_id, image=imagetk)
        else:
            self.image_id = self.create_image(0, 0, anchor="nw", image=imagetk)
        self.imgref = imagetk
