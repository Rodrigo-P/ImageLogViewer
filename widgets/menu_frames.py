from utils.display_functions import PROCESSING_FUNCTION
from utils.operations import is_number
from utils.style_formatting import *
from utils.constants import *
from utils.segmentation import (
    AsyncSegmentation,
)
from utils.classification import (
    ClassificationModule,
)

from customtkinter import (
    CTkOptionMenu,
    CTkCheckBox,
    CTkButton,
    CTkEntry,
    CTkFrame,
    CTkLabel,
)

from customtkinter import (
    StringVar,
    IntVar,
)

from typing import Dict, List, Callable

import json
import os


class LoadingMenuFrame(CTkFrame):
    def __init__(
        self, master, loading_function: Callable, data_type_list, *args, **kwargs
    ) -> None:
        super().__init__(master, *args, **kwargs)
        self.rowconfigure((0, 2), weight=1)
        self.rowconfigure((1), weight=5)
        self.columnconfigure((0), weight=1)

        self.data_title = CTkLabel(
            self,
            text="Poço ---",
            font=(FONT, 18),
            fg_color="#282F32",
            bg_color="#282F32",
            text_color="white",
        )

        self.loaded_data_widgets: Dict[str, List[CTkLabel]] = {}

        self.loaded_data_display = CTkFrame(self, **MENU_FRAME_STYLE)
        self._build_loaded_data_display(data_type_list)

        self.load_button = CTkButton(
            command=loading_function,
            master=self,
            text="LOAD",
            font=(FONT, 14),
            **BUTTON_FORMATTING,
        )

        self.data_title.pack(side=ctk.TOP, fill=ctk.X)
        self.load_button.pack(
            side=ctk.BOTTOM,
            fill=ctk.X,
            pady=MENU_BORDER_PAD["Y"],
            padx=MENU_BORDER_PAD["X"],
        )
        self.loaded_data_display.pack(
            side=ctk.TOP, fill=ctk.BOTH, expand=True, padx=MENU_BORDER_PAD["X"]
        )

    def _build_loaded_data_display(self, data_type_list):
        self.loaded_data_display.columnconfigure((0, 1), weight=1)
        self.loaded_data_display.rowconfigure((0, 1, 2), weight=1)

        self.loaded_data_widgets = {}

        for index, data_type in enumerate(data_type_list):
            data_info_label = CTkLabel(
                self.loaded_data_display,
                text=f"{data_type}",
                text_color="white",
                font=(FONT, 16),
                justify="left",
                anchor="w",
            )
            data_info_label.grid(row=index, column=0, sticky="w", padx=2)

            data_loaded_label = CTkLabel(
                self.loaded_data_display,
                text="Carregado",
                font=(FONT, 16),
                text_color="#EE3333",
                bg_color="#330000",
            )
            data_loaded_label.grid(row=index, column=1, sticky="e", padx=2)

            self.loaded_data_widgets[data_type] = [data_info_label, data_loaded_label]

        self.load_results = CTkLabel(
            self.loaded_data_display, text="---", font=(FONT, 16), text_color="white"
        )


class ProcessingMenuFrame(CTkFrame):
    def __init__(
        self,
        master,
        display_type,
        make_button_function: Callable,
        update_function: Callable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(master, *args, **kwargs)

        self.columnconfigure((0, 1, 2), weight=1)
        self.rowconfigure((0, 1), weight=1)

        self.processing_buttons: Dict[str, CTkButton] = {}
        process_types = PROCESSING_FUNCTION.keys()

        for index, name in enumerate(process_types):
            column = index % 3
            row = index // 3

            self.processing_buttons[name] = make_button_function(
                command=lambda button_type=name: update_function(button_type),
                master=self,
                column=column,
                row=row,
                text=name,
                padx=(
                    MENU_BORDER_PAD["X"] if column == 0 else 3,
                    MENU_BORDER_PAD["X"] if column == 2 else 3,
                ),
                pady=(
                    MENU_BORDER_PAD["Y"] if row == 0 else 3,
                    MENU_BORDER_PAD["Y"] if row == 1 else 3,
                ),
            )
        self.processing_buttons[display_type].configure(state=ctk.DISABLED)


class DataSelectionMenuFrame(CTkFrame):
    def __init__(
        self,
        master,
        data_type_list,
        display_state_changed_function: Callable,
        update_display_state_function: Callable,
        update_image_grid_function: Callable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(master, *args, **kwargs)
        self.columnconfigure(0, weight=9)
        self.columnconfigure(1, weight=1)
        self.rowconfigure((0, 1, 2), weight=1)

        self.checkboxes: List[CTkCheckBox] = []
        self.optionmenus: List[CTkOptionMenu] = []
        self.column_vars: List[StringVar] = []
        self.check_vars: List[IntVar] = []

        self.display_options = ["Col 1", "Col 2", "Col 3"]
        self.previous_display_options = ["", "", ""]

        last_row = len(data_type_list) - 1

        self._display_state_changed = display_state_changed_function
        self._update_display_state = update_display_state_function
        self._update_image_grid = update_image_grid_function

        for index, data_type in enumerate(data_type_list):
            self.check_vars.append(IntVar(self))

            self.checkboxes.append(
                CTkCheckBox(
                    self,
                    text=data_type,
                    variable=self.check_vars[index],
                    command=lambda *args, pos=index: self._display_check_flipped(
                        position=pos
                    ),
                    onvalue=1,
                    offvalue=0,
                    **CHECKBOX_STYLE,
                )
            )

            self.column_vars.append(StringVar(self))
            self.column_vars[index].trace_add(
                "write",
                lambda *args, pos=index: self._display_option_changed(
                    position=pos, args=args
                ),
            )
            self.optionmenus.append(
                CTkOptionMenu(
                    variable=self.column_vars[index],
                    master=self,
                    state=ctk.DISABLED,
                    values=self.display_options,
                    font=(FONT, 16),
                    **OPTIONSMENU_STYLE,
                )
            )

            self.checkboxes[index].grid(
                sticky=FULL_STICK,
                row=index,
                column=0,
                padx=(MENU_BORDER_PAD["X"], 4),
                pady=(
                    MENU_BORDER_PAD["Y"] if index == 0 else 5,
                    MENU_BORDER_PAD["Y"] if index == last_row else 5,
                ),
            )
            self.optionmenus[index].grid(
                row=index,
                column=1,
                padx=(4, MENU_BORDER_PAD["X"]),
                pady=(
                    MENU_BORDER_PAD["Y"] if index == 0 else 5,
                    MENU_BORDER_PAD["Y"] if index == last_row else 5,
                ),
            )

    def _display_check_flipped(self, position: int):
        state = self.check_vars[position].get()

        if state:
            self.optionmenus[position].configure(state=ctk.NORMAL)
        else:
            self.optionmenus[position].configure(state=ctk.DISABLED)

        self._display_state_changed()
        self._update_display_state()

    def _display_option_changed(self, position: int, **kwargs):
        new_value = self.column_vars[position].get()
        old_value = self.previous_display_options[position]

        if old_value != "" and new_value != "":
            if new_value in self.previous_display_options:
                conflict_index = self.previous_display_options.index(new_value)
                if conflict_index != position:

                    self.previous_display_options[conflict_index] = ""
                    self.column_vars[conflict_index].set(old_value)

            self._update_image_grid()

        self.previous_display_options[position] = new_value


class WindowingMenuFrame(CTkFrame):
    def __init__(
        self,
        master,
        get_state_function: Callable,
        flip_state_function: Callable,
        make_button_function: Callable,
        windowing_update_function: Callable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(master, *args, **kwargs)
        self.columnconfigure((0, 1), weight=1)
        self.rowconfigure((0, 1, 2), weight=1)

        self.windowing_update_function = windowing_update_function
        self.flip_state_function = flip_state_function
        self.get_state_function = get_state_function

        self.windowing_button = make_button_function(
            command=self.windowing_button_click,
            master=self,
            text="Custom Windowing",
            column=0,
            row=2,
            columnspan=2,
            padx=MENU_BORDER_PAD["X"],
            pady=(3, MENU_BORDER_PAD["Y"]),
        )

        self.windowing_entries: Dict[str, CTkEntry] = {}
        self.string_variables: Dict[str, StringVar] = {}

        for index, input_type in enumerate(["CENTER", "WIDTH"]):
            type_label = CTkLabel(
                master=self,
                font=(FONT, 16, "bold"),
                text_color="white",
                text=input_type,
                height=20,
            )

            type_label.grid(
                column=index,
                row=1,
                padx=(
                    MENU_BORDER_PAD["X"] if index == 0 else 4,
                    MENU_BORDER_PAD["X"] if index == 1 else 4,
                ),
                pady=3,
            )

            self.string_variables[input_type] = StringVar()
            self.windowing_entries[input_type] = CTkEntry(
                self,
                textvariable=self.string_variables[input_type],
                bg_color="#1D2325",
                fg_color="#1D2325",
                text_color="white",
                justify="center",
                corner_radius=0,
                border_width=1,
            )

            self.windowing_entries[input_type].grid(
                column=index,
                row=0,
                padx=(
                    MENU_BORDER_PAD["X"] if index == 0 else 4,
                    MENU_BORDER_PAD["X"] if index == 1 else 4,
                ),
                pady=(MENU_BORDER_PAD["Y"], 3),
            )
            self.windowing_entries[input_type].delete(0, ctk.END)

        self.windowing_entries["CENTER"].insert(0, "195")
        self.string_variables["CENTER"].trace_add(
            "write", lambda *args: self._update_custom_windowing()
        )

        self.windowing_entries["WIDTH"].insert(0, "50")
        self.string_variables["WIDTH"].trace_add(
            "write", lambda *args: self._update_custom_windowing()
        )

    def windowing_button_click(self):
        self.flip_state_function()
        self._update_custom_windowing()

    def _update_custom_windowing(self):
        if not self.get_state_function():
            return

        center = self.windowing_entries["CENTER"].get()
        width = self.windowing_entries["WIDTH"].get()

        if is_number(width) and is_number(center):
            self.windowing_update_function(float(center), float(width))


class MachineLearningMenuFrame(CTkFrame):
    def __init__(
        self,
        master,
        read_class_predictions_function: Callable,
        display_state_changed_function: Callable,
        flip_classification_function: Callable,
        flip_segmentation_function: Callable,
        class_prediction_setter: Callable,
        resistive_data_getter: Callable,
        make_button_function: Callable,
        package_path: str,
        imglog_name: str,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(master, *args, **kwargs)

        self._read_class_predictions = read_class_predictions_function
        self._display_state_changed = display_state_changed_function
        self._set_class_prediction = class_prediction_setter
        self._get_resistive_data = resistive_data_getter
        self.flip_classification = flip_classification_function
        self.flip_segmentation = flip_segmentation_function
        self._make_button = make_button_function
        self.package_path = package_path
        self.imglog_name = imglog_name

        self.columnconfigure((0, 1), weight=1)
        self.rowconfigure(0, weight=1)

        self.classification_functions = ["VGG16", "Annotated"]
        self.classification_process = None
        self.active_classification_jobs = []

        self.segmentation_functions = ["OTSU", "SAM", "SLIC"]
        self.segmentation_process: AsyncSegmentation = None
        self.active_segmentation_jobs = []

        self._build_menu_classification()
        self._build_menu_segmentation()

    def _build_menu_classification(self):
        self.classification_menu = CTkFrame(self, **MENU_FRAME_STYLE)
        self.classification_menu.grid(row=0, column=1, sticky=FULL_STICK, padx=1)
        self.classification_menu.columnconfigure(0, weight=1)
        self.classification_menu.rowconfigure((0, 1), weight=1)

        self.classification_var = StringVar(self)
        self.classification_var.set(self.classification_functions[0])
        self.classification_var.trace_add(
            "write", lambda *args: self._classification_option_changed()
        )

        self._classification_mode = self.classification_var.get()

        self.classification_optionmenu = CTkOptionMenu(
            values=self.classification_functions,
            variable=self.classification_var,
            master=self.classification_menu,
            font=(FONT, 16),
            **OPTIONSMENU_STYLE,
        )

        self.classification_button = self._make_button(
            command=self._classification_button_click,
            master=self.classification_menu,
            text="Classification",
            column=0,
            row=1,
            padx=MENU_BORDER_PAD["X"],
            pady=(3, MENU_BORDER_PAD["Y"]),
        )
        self.classification_optionmenu.grid(
            column=0,
            row=0,
            padx=MENU_BORDER_PAD["X"],
            pady=(MENU_BORDER_PAD["Y"], 3),
        )

    def _build_menu_segmentation(self):
        self.segmentation_menu = CTkFrame(self, **MENU_FRAME_STYLE)
        self.segmentation_menu.grid(row=0, column=0, sticky=FULL_STICK, padx=1)
        self.segmentation_menu.columnconfigure(0, weight=1)
        self.segmentation_menu.rowconfigure((0, 1), weight=1)

        self.segmentation_var = StringVar(self)
        self.segmentation_var.set(self.segmentation_functions[2])
        self.segmentation_var.trace_add(
            "write", lambda *args: self._segmentation_option_changed()
        )

        self._segmentation_mode = self.segmentation_var.get()

        self.segmentation_optionmenu = CTkOptionMenu(
            values=self.segmentation_functions,
            variable=self.segmentation_var,
            master=self.segmentation_menu,
            font=(FONT, 16),
            **OPTIONSMENU_STYLE,
        )

        self.classification_button = self._make_button(
            command=self._segmentation_button_click,
            master=self.segmentation_menu,
            text="Segmentation",
            column=0,
            row=1,
            padx=MENU_BORDER_PAD["X"],
            pady=(3, MENU_BORDER_PAD["Y"]),
        )
        self.segmentation_optionmenu.grid(
            column=0,
            row=0,
            padx=MENU_BORDER_PAD["X"],
            pady=(MENU_BORDER_PAD["Y"], 3),
        )

    def _classification_option_changed(self):
        self._classification_mode = self.classification_var.get()

        if self.classification_active:
            self.classification_active = False

            self._display_state_changed()

    def _classification_button_click(self):
        classification_active = self.flip_classification()

        if classification_active:
            if self._classification_mode == "Annotated":
                self._set_class_prediction(self._read_class_predictions())
            else:
                self._classification_run()

        self._display_state_changed()

    def _classification_run(self):
        cache_folder_path = (
            f"{self.package_path}/cache/{self.imglog_name}/{self._classification_mode}"
        )

        self.classification_process = ClassificationModule(
            self._classification_mode,
            self._get_resistive_data(),
            self.package_path,
            self.imglog_name,
        )
        self.classification_process.run()

        json_path = f"{cache_folder_path}/class_prediction.json"
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_data: Dict = json.load(json_file)

            self._set_class_prediction(
                {int(key): value for key, value in json_data.items()}
            )

    def _segmentation_option_changed(self):
        self._segmentation_mode = self.segmentation_var.get()

        if self.segmentation_active:
            self._segmentation_run()

            self._display_state_changed()

    def _segmentation_button_click(self):
        segmentation_active = self.flip_segmentation()

        if segmentation_active:
            self._segmentation_run()
        elif self.segmentation_process:
            self.segmentation_process.terminate()

        self._display_state_changed()

    def _segmentation_run(self):
        if self.segmentation_process != None:
            self.segmentation_process.terminate()

        cache_folder_path = (
            f"{self.package_path}/cache/{self.imglog_name}/{self._segmentation_mode}"
        )
        os.makedirs(cache_folder_path, exist_ok=True)

        self.segmentation_process = AsyncSegmentation(
            self._segmentation_mode,
            self._get_resistive_data(),
            self.package_path,
            self.imglog_name,
        )
        self.segmentation_process.start()


class PositionZoomMenu(CTkFrame):
    def __init__(
        self,
        master,
        previous_position_function: Callable,
        next_position_function: Callable,
        make_button_function: Callable,
        zoom_out_function: Callable,
        zoom_in_function: Callable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(master, *args, **kwargs)
        self.columnconfigure((0, 1, 2, 3), weight=1)
        self.rowconfigure((0, 1), weight=1)

        self.zoom_label = CTkLabel(
            master=self,
            font=(FONT, 18, "bold"),
            text_color="white",
            text="Zoom",
        )
        self.position_label = CTkLabel(
            master=self,
            font=(FONT, 18, "bold"),
            text_color="white",
            text="Position",
        )
        self.zoom_label.grid(
            row=0,
            column=0,
            columnspan=2,
            padx=MENU_BORDER_PAD["X"],
            pady=(4, 0),
            sticky=FULL_STICK,
        )
        self.position_label.grid(
            row=0,
            column=2,
            columnspan=2,
            padx=MENU_BORDER_PAD["X"],
            pady=(4, 0),
            sticky=FULL_STICK,
        )

        self.previous_button: CTkButton = make_button_function(
            master=self,
            command=previous_position_function,
            text="<",
            column=2,
            row=1,
            padx=(MENU_BORDER_PAD["X"], 4),
            pady=(0, MENU_BORDER_PAD["Y"]),
            bold_text=True,
        )
        self.next_button: CTkButton = make_button_function(
            master=self,
            command=next_position_function,
            text=">",
            column=3,
            row=1,
            padx=(4, MENU_BORDER_PAD["X"]),
            pady=(0, MENU_BORDER_PAD["Y"]),
            bold_text=True,
        )

        self.previous_button.configure(state=ctk.DISABLED)
        self.next_button.configure(state=ctk.DISABLED)

        self.zoom_out_button: CTkButton = make_button_function(
            master=self,
            command=zoom_out_function,
            text="-",
            column=1,
            row=1,
            padx=(4, MENU_BORDER_PAD["X"]),
            pady=(0, MENU_BORDER_PAD["Y"]),
            bold_text=True,
        )
        self.zoom_in_button: CTkButton = make_button_function(
            master=self,
            command=zoom_in_function,
            text="+",
            column=0,
            row=1,
            padx=(MENU_BORDER_PAD["X"], 4),
            pady=(0, MENU_BORDER_PAD["Y"]),
            bold_text=True,
        )
