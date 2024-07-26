from utils.style_formatting import BACKGROUND_COLOR
from utils.constants import FULL_STICK

from widgets.imagelog_canvas import ImagelogCanvas

from customtkinter import CTkCanvas, CTkFrame

from typing import Dict, List
from PIL import Image, ImageTk


class CanvasGridManagerFrame(CTkFrame):
    def __init__(
        self, root_class, number_of_columns, number_of_rows, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.canvas_grid: List[List[ImagelogCanvas]] = None
        self.images_grid: List[List[Image.Image]] = None

        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows
        self.root_class = root_class

        self._build_grid()

    def update_image(self, column, row, pil_image):
        canvas: ImagelogCanvas = self.canvas_grid[column][row]
        image_id = self.canvas_grid[column][row].image_id

        self.images_grid[column][row] = pil_image
        canvas.pil_image = pil_image

        imagetk = ImageTk.PhotoImage(image=pil_image)
        canvas.itemconfigure(image_id, image=imagetk)
        canvas.imgref = imagetk

        canvas.event_generate(
            "<Configure>", width=canvas.winfo_width(), height=canvas.winfo_height()
        )

    def _build_grid(self):
        self.canvas_grid = []
        self.images_grid = []

        for row in range(self.number_of_columns):
            self.images_grid.append([])
            self.canvas_grid.append([])

        for row in range(self.number_of_rows):
            for column in range(self.number_of_columns):
                canvas = ImagelogCanvas(
                    windowing_frame=self.root_class.windowing_menu,
                    background=BACKGROUND_COLOR,
                    root_class=self.root_class,
                    highlightthickness=0,
                    master=self,
                    relief="ridge",
                    manager=self,
                    row=row,
                    bd=0,
                )
                canvas.grid(row=row, column=column, padx=5, sticky=FULL_STICK)

                self.images_grid[column].append(self.root_class.blank_image_pil)
                self.canvas_grid[column].append(canvas)

    def hover_line_update_state(self, active_row, state):
        for column in range(self.number_of_columns):
            line_id = self.canvas_grid[column][active_row].hover_line_id
            canvas: CTkCanvas = self.canvas_grid[column][active_row]

            canvas.itemconfig(line_id, state=state)

    def hover_line_update_position(self, event, active_row):
        for column in range(self.number_of_columns):
            line_id = self.canvas_grid[column][active_row].hover_line_id
            canvas: CTkCanvas = self.canvas_grid[column][active_row]

            canvas.coords(line_id, 0, event.y, canvas.winfo_width(), event.y)

    def remake_grid(self, number_of_columns, number_of_rows):
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows

        self._clear_grid()
        self._build_grid()

    def _clear_grid(self):
        for column in self.canvas_grid:
            for canvas in column:
                canvas.grid_remove()
                canvas.destroy()

    def __del__(self):
        self._clear_grid()
