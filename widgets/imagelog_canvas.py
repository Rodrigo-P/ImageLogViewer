from utils.style_formatting import FONT
from utils.operations import is_number

from widgets.menu_frames import WindowingMenuFrame

from customtkinter import CTkCanvas
from PIL import Image, ImageTk


class ImagelogCanvas(CTkCanvas):
    def __init__(
        self,
        root_class,
        windowing_frame: WindowingMenuFrame,
        manager,
        row,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.windowing_frame = windowing_frame
        self.root_class = root_class
        self.manager = manager
        self.row_number = row
        self.pil_image = None
        self.last_x = None
        self.last_y = None

        self._build()

    def _build(self):
        self.pil_image = self.root_class.blank_image_pil
        self.image_id = self.create_image(
            0, 0, anchor="nw", image=ImageTk.PhotoImage(self.pil_image)
        )

        self.hover_line_id = self.create_line(0, 0, 4, 4, width=3, fill="red")

        self.itemconfig(self.hover_line_id, state="hidden")

        self.bind("<Configure>", self._resizer)

        self.bind("<ButtonRelease-1>", self._b1_release)
        self.bind("<B1-Motion>", self._b1_hover)
        self.bind("<Button-1>", self._b1_press)
        self.bind("<Motion>", self._motion)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)

        if (
            self.root_class.resistive_data is not None
            and self.root_class.resistive_data["class_prediction"] is not None
        ):
            self.text_id = self.create_text(
                5,
                2,
                font=(FONT, 12, "bold"),
                justify="left",
                state="hidden",
                fill="grey16",
                anchor="nw",
                text="Test",
            )

            self.rectangle_id = self.create_rectangle(
                (10, 10, 30, 30),
                outline="azure3",
                state="hidden",
                fill="azure3",
            )

            self.tag_lower(self.rectangle_id, self.text_id)

    def _enter(self, event):
        if not self.root_class.block_hover_line:
            self.manager.hover_line_update_state(self.row_number, "normal")

    def _leave(self, event):
        self.manager.hover_line_update_state(self.row_number, "hidden")

    def _motion(self, event):
        self.manager.hover_line_update_position(event, self.row_number)
        self.root_class.update_ruler_position(event, self.row_number)

    def _b1_hover(self, event):
        if self.root_class.custom_windowing_active:
            if self.last_y != None:
                y_difference = self.last_y - event.y
                x_difference = self.last_x - event.x

                center = self.windowing_frame.windowing_entries["CENTER"].get()
                width = self.windowing_frame.windowing_entries["WIDTH"].get()
                if not (is_number(center) and is_number(width)):
                    return
                else:
                    center = float(center) + (y_difference) / 5
                    width = float(width) + (x_difference) / 5

                self.windowing_frame.string_variables["CENTER"].set(f"{center:.1f}")
                self.windowing_frame.string_variables["WIDTH"].set(f"{width:.1f}")

            self.last_y = event.y
            self.last_x = event.x

    def _b1_press(self, event):
        if self.root_class.custom_windowing_active:
            self.manager.hover_line_update_state(self.row_number, "hidden")
            self.root_class.root_app.configure(cursor="fleur")
            self.root_class.block_hover_line = True

    def _b1_release(self, event):
        if self.root_class.custom_windowing_active:
            self.manager.hover_line_update_state(self.row_number, "normal")
            self.root_class.root_app.configure(cursor="arrow")
            self.root_class.block_hover_line = False

            self.last_x = None
            self.last_y = None

    def _resizer(self, event):
        if self.pil_image == None:
            return

        image_id = self.image_id
        height = event.height
        width = event.width

        side = min(height, width)
        # self.side = side

        resized_image = self.pil_image.resize((side, side), Image.Resampling.LANCZOS)
        imagetk = ImageTk.PhotoImage(resized_image)
        self.itemconfigure(image_id, image=imagetk)
        self.imgref = imagetk
        self.moveto(image_id, (width - side) // 2, (height - side) // 2)

        # self.manager._view_port_update_height(side)
