import sys, os, dotenv

dotenv.load_dotenv()
PACKAGEPATH = os.getenv("PACKAGEPATH")
sys.path.append(PACKAGEPATH)

from utils.style_formatting import *
from utils.constants import *

from widgets.imagelog_canvas import ImagelogCanvas
from widgets.canvas_grid import CanvasGridManagerFrame
from widgets.position_canvases import (
    RulerCanvas,
    MiniatureLogCanvas,
    StructureViewCanvas,
)
from widgets.menu_frames import *

from utils.display_functions import (
    PROCESSING_FUNCTION,
    make_windowing_function,
)
from utils.segmentation import (
    AsyncSegmentation,
)
from utils.classification import (
    ClassificationModule,
)
from utils.imagelog_functions import (
    draw_rectangle_border,
    align_acoustic_signal,
    load_image_log_csv,
    image_from_signal,
    float32_to_pil,
)

from customtkinter import (
    CTkLabel,
    CTkCanvas,
    CTkButton,
    CTkOptionMenu,
    CTkFrame,
    CTkEntry,
    CTkCheckBox,
)
from customtkinter import StringVar, IntVar
from customtkinter import filedialog

from tkinter.messagebox import showinfo


from skimage.color import label2rgb
from typing import Dict, List
from PIL import ImageTk


import customtkinter as ctk
import numpy as np
import json
import math


class ImageViewer:
    def __init__(self) -> None:
        self.resistive_data = None
        self.acoustic_data = None
        self.static_data = None

        self.data_type_list = ["Resistivo Dinâmico", "Resistivo Estático", "Acústico"]

        self.classification_functions = ["VGG16", "Annotated"]
        self.segmentation_functions = ["OTSU", "SAM", "SLIC"]
        self.classification_active = False
        self.segmentation_active = False

        self.custom_windowing_active = False
        self.last_x = None
        self.last_y = None

        self.number_of_columns = 0
        self.number_of_rows = 2
        self.tale_size = 360

        self.current_position = 0

        # App Frame
        self.root_app = ctk.CTk()
        self.root_app.geometry("1280x900")
        self.root_app.title("IMLOG Viewer")
        self._try_maximize_window()

        self.root_app.configure(bg_color=BACKGROUND_COLOR)
        self.root_app.configure(fg_color=BACKGROUND_COLOR)

        # Grid configure
        self.root_app.columnconfigure(1, weight=1)
        self.root_app.rowconfigure(0, weight=1)

        # Keyboard binding
        self.root_app.bind("<Left>", lambda event: self._previous_position())
        self.root_app.bind("<Right>", lambda event: self._next_position())

        # Default blank image
        self.blank_image_pil = float32_to_pil(
            np.ones((self.tale_size, self.tale_size, 3)) * 0.5
        )
        self.blank_image_tk = ImageTk.PhotoImage(image=self.blank_image_pil)
        self.display_type = "Colorido"

        self._setup_frames()
        self._setup_widgets()
        self.block_hover_line = False

        # Run
        self._load_data_folder()
        self.root_app.mainloop()

    def _try_maximize_window(self):
        try:
            self.root_app.attributes("-zoomed", True)
        except:
            try:
                self.root_app.state("zoomed")
            except:
                pass

    def _setup_frames(self):
        # Main Frames

        self.menu_frame = CTkFrame(
            self.root_app, width=300, bg_color="#2D3237", fg_color="#2D3237"
        )
        # self.image_display_frame = CTkFrame(self.root_app, corner_radius=0)
        self.image_display_frame = CanvasGridManagerFrame(
            number_of_columns=self.number_of_columns,
            number_of_rows=self.number_of_rows,
            fg_color=BACKGROUND_COLOR,
            bg_color=BACKGROUND_COLOR,
            master=self.root_app,
            root_class=self,
        )
        self.position_view_frame = CTkFrame(self.root_app, width=144, corner_radius=0)

        root_grid_config = {"sticky": VERTICAL_STICK, "row": 0}
        self.position_view_frame.grid(column=0, **root_grid_config)
        self.image_display_frame.grid(column=1, **root_grid_config)
        self.menu_frame.grid(column=2, padx=1, pady=1, **root_grid_config)

        self.image_display_frame.columnconfigure((0, 1, 2), weight=1)
        self.image_display_frame.rowconfigure((0, 1), weight=1)

        self.position_view_frame.columnconfigure((0, 1), weight=1)
        self.position_view_frame.rowconfigure((0), weight=1)

        self.image_display_frame.grid_propagate(False)
        # self.position_view_frame.grid_propagate(False)
        self.menu_frame.pack_propagate(False)
        self.menu_frame.grid_propagate(False)

        self.image_display_frame.bind(
            "<Configure>",
            self._update_display_frame,
        )
        self.image_display_frame.bind(
            "<Motion>",
            lambda e: self.update_ruler_position(e, 0),
        )

    def _setup_widgets(self):
        self._build_position_viewer()
        self._build_control_menu()
        self._pack_control_menu()

    def _build_control_menu(self):
        self.machine_learning_menu = MachineLearningMenuFrame(
            read_class_predictions_function=self._read_class_predictions,
            flip_classification_function=self._flip_classification_state,
            display_state_changed_function=self._display_state_changed,
            flip_segmentation_function=self._flip_segmentation_state,
            class_prediction_setter=self.set_class_prediction,
            resistive_data_getter=self.get_resistive_data,
            make_button_function=self._make_button,
            package_path=PACKAGEPATH,
            imglog_name="imagelog",
            bg_color="#2D3237",
            fg_color="#2D3237",
            master=self.menu_frame,
        )
        self.data_selection_menu = DataSelectionMenuFrame(
            master=self.menu_frame,
            data_type_list=self.data_type_list,
            display_state_changed_function=self._display_state_changed,
            update_display_state_function=self._update_display_state,
            update_image_grid_function=self._update_image_grid,
            **MENU_FRAME_STYLE,
        )
        self.position_zoom_menu = PositionZoomMenu(
            previous_position_function=self._previous_position,
            next_position_function=self._next_position,
            make_button_function=self._make_button,
            zoom_out_function=self._zoom_out,
            zoom_in_function=self._zoom_in,
            master=self.menu_frame,
            **MENU_FRAME_STYLE,
        )
        self.image_processing_menu = ProcessingMenuFrame(
            display_type=self.display_type,
            make_button_function=self._make_button,
            update_function=self._update_image_processing,
            master=self.menu_frame,
            **MENU_FRAME_STYLE,
        )
        self.windowing_menu = WindowingMenuFrame(
            windowing_update_function=self._update_custom_windowing,
            flip_state_function=self._flip_custom_windowing_state,
            get_state_function=self._get_custom_windowing_state,
            make_button_function=self._make_button,
            master=self.menu_frame,
            **MENU_FRAME_STYLE,
        )
        self.imagelog_loading_menu = LoadingMenuFrame(
            loading_function=self._load_data_folder,
            data_type_list=self.data_type_list,
            master=self.menu_frame,
            **MENU_FRAME_STYLE,
        )
        self.caption_menu = None

    def _update_control_menu(self):
        if self.caption_menu != None:
            self.caption_menu.destroy()
            del self.caption_menu
            self.caption_menu = None

        if self.bbox_label_color_dictionary != None:
            self._build_menu_caption()

        self._pack_control_menu()

    def _update_position_viewer(self):
        if self.structure_view != None:
            self.structure_view.destroy()
            del self.structure_view
            self.structure_view = None

        if self.bbox_label_color_dictionary != None:
            self.structure_view = StructureViewCanvas(
                master=self.position_view_frame,
                root_class=self,
                bbox_label_color_dictionary=self.bbox_label_color_dictionary,
                bbox_data=self.resistive_data["bbox_data"],
                highlightthickness=0,
                width=40,
                bd=0,
            )

        self._pack_position_viewer()

    def _pack_position_viewer(self):
        self.miniature_canvas.pack_forget()
        self.ruler_canvas.pack_forget()

        self.miniature_canvas.pack(side=ctk.LEFT, fill=ctk.Y, expand=True, padx=(0, 1))

        if self.structure_view != None:
            self.structure_view.pack(side=ctk.LEFT, fill=ctk.Y, expand=True)

        self.ruler_canvas.pack(side=ctk.LEFT, fill=ctk.Y, expand=True, padx=(1, 0))

    def _build_position_viewer(self):
        self.miniature_canvas = MiniatureLogCanvas(
            master=self.position_view_frame,
            background=BACKGROUND_COLOR,
            highlightthickness=0,
            root_class=self,
            relief="ridge",
            width=20,
            bd=0,
        )
        self.ruler_canvas = RulerCanvas(
            master=self.position_view_frame,
            background=BACKGROUND_COLOR,
            highlightthickness=0,
            root_class=self,
            relief="ridge",
            width=70,
            bd=0,
        )
        self.structure_view = None
        self.miniature_canvas.pack(side=ctk.LEFT, fill=ctk.Y, expand=True)
        self.ruler_canvas.pack(side=ctk.LEFT, fill=ctk.Y, expand=True)

    def _pack_control_menu(self):
        self.image_processing_menu.pack_forget()
        self.imagelog_loading_menu.pack_forget()
        self.machine_learning_menu.pack_forget()
        self.data_selection_menu.pack_forget()
        self.position_zoom_menu.pack_forget()
        self.windowing_menu.pack_forget()

        self.position_zoom_menu.pack(
            side=ctk.BOTTOM, fill=ctk.X, padx=3, pady=(0, 3), expand=False
        )
        if self.caption_menu != None:
            self.caption_menu.pack(
                side=ctk.BOTTOM, fill=ctk.X, padx=3, pady=(0, 3), expand=False
            )
        self.machine_learning_menu.pack(
            side=ctk.BOTTOM, fill=ctk.X, padx=2, pady=(0, 3), expand=False
        )
        self.windowing_menu.pack(
            side=ctk.BOTTOM, fill=ctk.X, padx=3, pady=(0, 3), expand=False
        )
        self.data_selection_menu.pack(
            side=ctk.BOTTOM, fill=ctk.X, padx=3, pady=(0, 3), expand=False
        )
        self.image_processing_menu.pack(
            side=ctk.BOTTOM, fill=ctk.X, padx=3, pady=(0, 3), expand=False
        )
        self.imagelog_loading_menu.pack(
            side=ctk.BOTTOM, fill=ctk.BOTH, padx=3, pady=3, expand=True
        )

    def _update_image_processing(self, method: str):
        for button in self.image_processing_menu.processing_buttons.values():
            button.configure(state=ctk.NORMAL)

        self.image_processing_menu.processing_buttons[method].configure(
            state=ctk.DISABLED
        )

        self.display_type = method

        self._update_image_grid()

    def _make_button(
        self,
        master,
        command,
        text,
        row,
        column,
        padx=4,
        pady=4,
        font_size=14,
        bold_text=False,
        **grid_kwargs,
    ):
        if bold_text:
            font = (FONT, font_size, "bold")
        else:
            font = (FONT, font_size)

        new_button = CTkButton(
            command=command,
            master=master,
            text=text,
            font=font,
            **BUTTON_FORMATTING,
        )
        new_button.grid(
            row=row,
            column=column,
            padx=padx,
            pady=pady,
            sticky=SIDE_STICK,
            **grid_kwargs,
        )

        return new_button

    def _display_state_changed(self):
        self.number_of_columns = 0
        for index in range(3):
            if self.data_selection_menu.check_vars[index].get():
                self.number_of_columns += 1

        self.display_options = [f"Col {i+1}" for i in range(self.number_of_columns)]

        col_pos = 1
        for index in range(3):
            self.data_selection_menu.optionmenus[index].configure(
                values=self.display_options
            )
            self.data_selection_menu.previous_display_options[index] = ""

        for index in range(3):
            if self.data_selection_menu.check_vars[index].get():
                self.data_selection_menu.column_vars[index].set(f"Col {col_pos}")
                col_pos += 1
            else:
                self.data_selection_menu.column_vars[index].set("")

        self.image_display_frame.columnconfigure((0, 1, 2, 3), weight=0)
        if self.number_of_columns > 0:
            self.image_display_frame.columnconfigure(
                [i for i in range(self.number_of_columns)], weight=1
            )

        if self.custom_windowing_active:
            self.image_display_frame.columnconfigure(self.number_of_columns, weight=1)
            self.windowing_column_number = self.number_of_columns
            self.number_of_columns += 1

        if self.segmentation_active:
            self.image_display_frame.columnconfigure(self.number_of_columns, weight=1)
            self.segment_column_number = self.number_of_columns
            self.number_of_columns += 1

        self._update_display_state()

    def _update_display_state(self):
        self._update_display_frame(self.image_display_frame)
        self.image_display_frame.remake_grid(
            self.number_of_columns, self.number_of_rows
        )
        self._update_image_grid()

    def _build_menu_caption(self):
        self.caption_menu = CTkFrame(self.menu_frame, **MENU_FRAME_STYLE)

        number_of_classes = len(self.bbox_label_color_dictionary)

        title_label = CTkLabel(
            self.caption_menu, text="Legenda", font=(FONT, 16), text_color="white"
        )

        if number_of_classes == 1:
            self.caption_menu.columnconfigure((0), weight=1)
            title_label.grid(column=0, row=0)
        else:
            self.caption_menu.columnconfigure((0, 1), weight=1)

            title_label.grid(column=0, row=0, columnspan=2)

        number_of_rows = math.ceil(number_of_classes / 2)

        for row in range(number_of_rows):
            self.caption_menu.rowconfigure(row, weight=1)

        index = 0
        for class_key, color in self.bbox_label_color_dictionary.items():
            if self.bbox_class_dictionary == None:
                text = str(class_key)
            else:
                text = self.bbox_class_dictionary[class_key]

            # The 'label' is a disabled button because CTkLabel doesn't have a border
            class_label = CTkButton(
                self.caption_menu,
                text_color_disabled="black",
                border_color=color["hex"],
                font=(FONT, 12, "bold"),
                state=ctk.DISABLED,
                fg_color="gray80",
                corner_radius=0,
                border_width=4,
                text=text,
            )

            class_label.grid(
                column=(index % 2),
                row=1 + (index // 2),
                padx=(
                    MENU_BORDER_PAD["X"] if (index % 2) == 0 else 4,
                    MENU_BORDER_PAD["X"] if (index % 2) == 1 else 4,
                ),
                pady=(0, MENU_BORDER_PAD["Y"]),
            )
            index += 1

    def get_resistive_data(self):
        return self.resistive_data

    def set_class_prediction(self, class_data_dictionary):
        self.resistive_data["class_prediction"] = class_data_dictionary

    def _previous_position(self):
        if self.resistive_data == None or self.current_position == 0:
            return

        self.position_zoom_menu.next_button.configure(state=ctk.NORMAL)
        self.current_position -= 1

        if self.current_position == 0:
            self.position_zoom_menu.previous_button.configure(state=ctk.DISABLED)

        self.ruler_canvas.update_values(
            depth_values=self.resistive_data["depth"],
            current_position=self.current_position,
            number_of_rows=self.number_of_rows,
        )
        self.miniature_canvas.update_image()
        self._update_image_grid()

    def _next_position(self):
        if self.resistive_data == None or self.current_position == self.max_position:
            return

        self.position_zoom_menu.previous_button.configure(state=ctk.NORMAL)
        self.current_position += 1

        if self.current_position == self.max_position:
            self.position_zoom_menu.next_button.configure(state=ctk.DISABLED)

        self.ruler_canvas.update_values(
            depth_values=self.resistive_data["depth"],
            current_position=self.current_position,
            number_of_rows=self.number_of_rows,
        )
        self.miniature_canvas.update_image()
        self._update_image_grid()

    def _draw_bbox(self, label, image, width=4):
        for index, bbox in enumerate(label["boxes"]):
            color = self.bbox_label_color_dictionary[label["labels"][index]][
                "rgb_float"
            ]

            x0 = int((bbox[0] - (bbox[2] / 2)))
            x1 = int((bbox[0] + (bbox[2] / 2)))
            y0 = int((bbox[1] - (bbox[3] / 2)))
            y1 = int((bbox[1] + (bbox[3] / 2)))

            draw_rectangle_border(image, x0, x1, y0, y1, color, width)

    def _update_image_column(
        self, input_data, column, processing_function, draw_bbox=True, show_class=True
    ):
        for row in range(self.number_of_rows):
            position = self.current_position + row

            window_slice = processing_function(
                input_data, self.tale_size * position, self.tale_size * (position + 1)
            )

            if draw_bbox:
                self._draw_bbox(
                    self.resistive_data["bbox_data"][position], window_slice
                )

            if (
                show_class
                and self.classification_active
                and self.resistive_data["class_prediction"] is not None
            ):
                self._update_class_display(column, row, position)

            self.image_display_frame.update_image(
                column, row, float32_to_pil(window_slice)
            )

    def _stop_segment_jobs(self, draw_bbox=False):
        # run twice in case of race condition
        for i in range(2):
            for job_id in self.machine_learning_menu.active_segmentation_jobs:
                try:
                    self.root_app.after_cancel(job_id)
                except:
                    pass

        self.machine_learning_menu.active_segmentation_jobs = []

    def _update_segment_image_column(self, draw_bbox=False):
        self._stop_segment_jobs()

        for row in range(self.number_of_rows):
            self.image_display_frame.update_image(
                self.segment_column_number, row, self.blank_image_pil
            )
            position = self.current_position + row

            # Create image update job and add to list
            self.machine_learning_menu.active_segmentation_jobs.append(
                self.root_app.after(
                    100,
                    self._update_segment_image,
                    position,
                    row,
                )
            )

    def _update_segment_image(self, position, row, draw_bbox=False):
        cache_file_path = f"{PACKAGEPATH}/cache/{self.imglog_name}/{self.machine_learning_menu._segmentation_mode}/{position}.npy"

        if not self.segmentation_active:
            return

        if not os.path.exists(cache_file_path):
            self.machine_learning_menu.active_segmentation_jobs[row] = (
                self.root_app.after(
                    10000,
                    self._update_segment_image,
                    position,
                    row,
                    draw_bbox,
                )
            )
            return

        segment_labels = np.load(cache_file_path)
        image_array = PROCESSING_FUNCTION["Cavidade"](
            loaded_data=self.resistive_data,
            start=self.tale_size * position,
            end=self.tale_size * (position + 1),
            rgb=False,
        )
        # output_image = label2rgb(label=segment_labels, image=image_array, bg_label=1)
        output_image = label2rgb(
            label=segment_labels,
            image=image_array,
            colors=[x["rgb_float"] for x in DISTINCT_COLORS_DICT],
            bg_label=0,
        )

        if draw_bbox:
            self._draw_bbox(self.resistive_data["bbox_data"][position], output_image)

        self.image_display_frame.update_image(
            self.segment_column_number, row, float32_to_pil(output_image)
        )

    def _update_image_grid(self):
        if self.data_selection_menu.check_vars[0].get():
            column = int(self.data_selection_menu.column_vars[0].get()[-1]) - 1
            self._update_image_column(
                processing_function=PROCESSING_FUNCTION[self.display_type],
                input_data=self.resistive_data,
                column=column,
            )

        if self.data_selection_menu.check_vars[1].get():
            column = int(self.data_selection_menu.column_vars[1].get()[-1]) - 1
            self._update_image_column(
                processing_function=PROCESSING_FUNCTION[self.display_type],
                input_data=self.static_data,
                column=column,
            )

        if self.data_selection_menu.check_vars[2].get():
            column = int(self.data_selection_menu.column_vars[2].get()[-1]) - 1
            self._update_image_column(
                processing_function=PROCESSING_FUNCTION[self.display_type],
                input_data=self.acoustic_data,
                column=column,
            )

        if self.custom_windowing_active:
            self._update_image_column(
                processing_function=PROCESSING_FUNCTION["Custom Window"],
                column=self.windowing_column_number,
                input_data=self.resistive_data,
                show_class=False,
                draw_bbox=False,
            )

        if self.segmentation_active:
            self._update_segment_image_column()

    def _update_class_display(self, column, row, position):
        canvas: ImagelogCanvas = self.image_display_frame.canvas_grid[column][row]
        rectangle_id = canvas.rectangle_id
        text_id = canvas.text_id

        canvas.itemconfigure(
            text=self.resistive_data["class_prediction"][position],
            tagOrId=text_id,
            state="normal",
        )
        canvas.itemconfigure(
            tagOrId=rectangle_id,
            state="normal",
        )
        canvas.coords(rectangle_id, canvas.bbox(text_id))

    def _load_acoustic_data(self, position=2):
        self.acoustic_data = load_image_log_csv(f"{self.data_folder_path}/ACUSTICO.csv")
        output_text = f"\nDim: {self.acoustic_data['signal'].shape}"

        align_acoustic_signal(self.acoustic_data, self.resistive_data)
        value_for_flip = np.min(self.acoustic_data["signal"]) + np.max(
            self.acoustic_data["signal"]
        )
        self.acoustic_data["signal"] = value_for_flip - self.acoustic_data["signal"]
        del self.acoustic_data["mask"]

        self.acoustic_data["equalized"] = image_from_signal(
            self.acoustic_data["signal"]
        )

        self.data_selection_menu.checkboxes[position].configure(state=ctk.NORMAL)
        self.data_selection_menu.optionmenus[position].configure(state=ctk.NORMAL)
        self.data_selection_menu.check_vars[position].set(1)

        return output_text

    def _update_display_frame(self, e):
        display_width = (self.menu_frame.winfo_height() // self.number_of_rows) + 16
        display_width *= self.number_of_columns
        self.image_display_frame.configure(width=display_width)

    def _disable_data_type(self, position: int):
        self.data_selection_menu.checkboxes[position].configure(state=ctk.DISABLED)
        self.data_selection_menu.optionmenus[position].configure(state=ctk.DISABLED)
        self.data_selection_menu.check_vars[position].set(0)

    def update_ruler_position(self, event, row):
        y = event.y + (row * self.image_display_frame.canvas_grid[0][0].winfo_height())

        self.ruler_canvas.coords(self.ruler_canvas.ruler_line_id, 0, y, 50, y)

    def _flip_custom_windowing_state(self):
        self.custom_windowing_active = not self.custom_windowing_active
        self._display_state_changed()

    def _flip_classification_state(self):
        self.classification_active = not self.classification_active
        return self.classification_active

    def _flip_segmentation_state(self):
        self.segmentation_active = not self.segmentation_active
        return self.segmentation_active

    def _get_custom_windowing_state(self):
        return self.custom_windowing_active

    def _update_custom_windowing(self, center, width):
        PROCESSING_FUNCTION["Custom Window"] = make_windowing_function(center, width)

        self._update_image_column(
            processing_function=PROCESSING_FUNCTION["Custom Window"],
            column=self.windowing_column_number,
            input_data=self.resistive_data,
            draw_bbox=False,
        )

    def _zoom_in(self):
        self.number_of_rows -= 1
        self._update_display_rows()

    def _zoom_out(self):
        self.number_of_rows += 1
        self._update_display_rows()

    def _update_display_rows(self):
        if self.number_of_rows == MAX_ROWS:
            self.position_zoom_menu.zoom_out_button.configure(state=ctk.DISABLED)
        else:
            self.position_zoom_menu.zoom_out_button.configure(state=ctk.NORMAL)

        if self.number_of_rows == MIN_ROWS:
            self.position_zoom_menu.zoom_in_button.configure(state=ctk.DISABLED)
        else:
            self.position_zoom_menu.zoom_in_button.configure(state=ctk.NORMAL)

        self.image_display_frame.rowconfigure([i for i in range(MAX_ROWS)], weight=0)
        if self.number_of_rows > 0:
            self.image_display_frame.rowconfigure(
                [i for i in range(self.number_of_rows)], weight=1
            )

        self.max_position = self.resistive_data["signal"].shape[0] // self.tale_size
        self.max_position -= self.number_of_rows

        self._update_display_state()
        self.ruler_canvas.update_values(
            depth_values=self.resistive_data["depth"],
            current_position=self.current_position,
            number_of_rows=self.number_of_rows,
        )
        self.miniature_canvas.update_image()

    def set_loaded_data_display(self, position, output_text):
        # Data info
        self.imagelog_loading_menu.loaded_data_widgets[self.data_type_list[position]][
            0
        ].configure(text=f"{self.data_type_list[position]}{output_text}")

        # Colored label
        if output_text == "":
            self.imagelog_loading_menu.loaded_data_widgets[
                self.data_type_list[position]
            ][1].configure(
                text_color="#EE3333",
                bg_color="#330000",
            )
        else:
            self.imagelog_loading_menu.loaded_data_widgets[
                self.data_type_list[position]
            ][1].configure(
                text_color="#33EE33",
                bg_color="#003300",
            )

    def _load_imlog(self, file_name: str, position: int):
        imlog_data = load_image_log_csv(f"{self.data_folder_path}/{file_name}.csv")
        imlog_data["equalized"] = image_from_signal(imlog_data["signal"])

        output_text = f"\nDim: {imlog_data['signal'].shape}"

        self.data_selection_menu.checkboxes[position].configure(state=ctk.NORMAL)
        self.data_selection_menu.optionmenus[position].configure(state=ctk.NORMAL)
        self.data_selection_menu.check_vars[position].set(1)

        return imlog_data, output_text

    def _load_data_folder(self):
        path_input = filedialog.askdirectory(initialdir=f"{PACKAGEPATH}/data")

        # Invalid path
        if path_input == () or path_input == None or path_input == "":
            return
        # No data found
        if not os.path.exists(f"{path_input}/IMLOG.csv"):
            return

        for i in range(len(self.data_type_list)):
            self.set_loaded_data_display(i, "")

        self.imglog_name = path_input[path_input.rfind("/") + 1 :]
        self.imagelog_loading_menu.data_title.configure(text=f"Poço {self.imglog_name}")
        self.machine_learning_menu.imglog_name = self.imglog_name
        self.data_folder_path = path_input
        self.number_of_rows = 2

        self.root_app.update()

        self.resistive_data, output_text = self._load_imlog("IMLOG", 0)
        self.tale_size = self.resistive_data["signal"].shape[1]

        self.set_loaded_data_display(0, output_text)

        if os.path.exists(f"{self.data_folder_path}/ESTATICO.csv"):
            self.static_data, output_text = self._load_imlog("ESTATICO", 1)

        else:
            self._disable_data_type(1)
            self.static_data = None
            output_text = ""
        self.set_loaded_data_display(1, output_text)

        if os.path.exists(f"{self.data_folder_path}/ACUSTICO.csv"):
            output_text = self._load_acoustic_data()

        else:
            self._disable_data_type(2)
            self.acoustic_data = None
            output_text = ""
        self.set_loaded_data_display(2, output_text)

        self.max_position = self.resistive_data["signal"].shape[0] // self.tale_size
        self.max_position -= self.number_of_rows

        self.resistive_data["bbox_data"] = [
            self._read_bbox_data(pos)
            for pos in range(self.max_position + self.number_of_rows)
        ]
        self._compile_bbox_data()

        self.resistive_data["class_prediction"] = None

        self.position_zoom_menu.previous_button.configure(state=ctk.DISABLED)
        self.position_zoom_menu.next_button.configure(state=ctk.NORMAL)
        self.current_position = 0

        self.miniature_canvas.event_generate(
            height=self.miniature_canvas.winfo_height(),
            width=self.miniature_canvas.winfo_width(),
            sequence="<Configure>",
        )

        self._display_state_changed()

        self.ruler_canvas.event_generate(
            height=self.ruler_canvas.winfo_height(),
            width=self.ruler_canvas.winfo_width(),
            sequence="<Configure>",
        )
        self._update_control_menu()
        self._update_position_viewer()

    def _compile_bbox_data(self):
        flat_bbox_data = [
            label
            for bbox_dict in self.resistive_data["bbox_data"]
            for label in bbox_dict["labels"]
        ]
        if len(flat_bbox_data) == 0:
            self.bbox_label_color_dictionary = None
            return

        bbox_labels_set = set(flat_bbox_data)
        labels_list = list(bbox_labels_set)
        labels_list.sort()

        self.bbox_label_color_dictionary = {}
        for index, class_value in enumerate(labels_list):
            self.bbox_label_color_dictionary[class_value] = DISTINCT_COLORS_DICT[index]

        class_dictionary_path = (
            f"{PACKAGEPATH}/data/{self.imglog_name}/labels/class_names.json"
        )
        if not os.path.exists(class_dictionary_path):
            self.bbox_class_dictionary = None
        else:
            self.bbox_class_dictionary = {}
            with open(class_dictionary_path, "r", encoding="utf-8") as json_file:
                bbox_class_dictionary: Dict[int, str] = json.load(json_file)

            for label in labels_list:
                if not str(label) in bbox_class_dictionary:
                    print(f"Missing label name for {label}")
                    self.bbox_class_dictionary[label] = str(label)
                else:
                    self.bbox_class_dictionary[label] = bbox_class_dictionary[
                        str(label)
                    ]

    def _read_bbox_data(self, position: int):
        label_path = f"{PACKAGEPATH}/data/{self.imglog_name}/labels/{position:06d}.txt"
        target_data = {"boxes": [], "labels": []}

        if not os.path.exists(label_path):
            return target_data

        text_file = open(label_path)

        for line in text_file:
            line = line.rstrip()
            value_list = line.split()
            class_index = value_list.pop(0)

            target_data["labels"].append(int(class_index))
            target_data["boxes"].append(
                [float(el) * self.tale_size for el in value_list]
            )

        target_data["labels"] = np.array(target_data["labels"])
        target_data["boxes"] = np.array(target_data["boxes"])

        return target_data

    def _read_class_predictions(self):
        json_path = f"{PACKAGEPATH}/data/{self.imglog_name}/class_data.json"

        if not os.path.exists(json_path):
            self.classification_active = False
            return None

        with open(json_path, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)

            return {int(key): value for key, value in json_data["prediction"].items()}



viewer_app = ImageViewer()
