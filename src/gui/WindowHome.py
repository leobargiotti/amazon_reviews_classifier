import tkinter.messagebox
import customtkinter
from functools import partial

from .WindowStatistics import WindowStatistics

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue", "green", "dark-blue"


class WindowHome(customtkinter.CTk):

    def __init__(self, array_classifiers, array_name_classifiers):
        super().__init__()

        self.title("Reviews Classifier")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.classifiers = array_classifiers
        self.name_classifiers = array_name_classifiers

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_buttons = customtkinter.CTkFrame(master=self,
                                                    width=180,
                                                    corner_radius=0)
        self.frame_buttons.grid(row=0, column=0, sticky="nswe")

        self.frame_button_classify = customtkinter.CTkFrame(master=self)
        self.frame_button_classify.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_buttons ============
        self.frame_buttons.grid_rowconfigure(5, weight=1)
        self.frame_buttons.grid_rowconfigure(11, minsize=25)

        self.label_settings = customtkinter.CTkLabel(master=self.frame_buttons,
                                                     text="Reviews Classifier",
                                                     font=("Roboto Medium", -16))
        self.label_settings.grid(row=0, column=0, pady=10, padx=10)

        self.button_text = ["Statistics"]
        self.button_event = [partial(self.button_event, 0)]
        self.buttons = []

        for index in range(len(self.button_event)):
            self.buttons.append(customtkinter.CTkButton(master=self.frame_buttons,
                                                        text=self.button_text[index],
                                                        border_width=2,
                                                        fg_color=None,
                                                        command=self.button_event[index]))
            self.buttons[index].grid(row=index + 1, column=0, columnspan=1, pady=20, padx=20, sticky="we")

        self.label_appearance = customtkinter.CTkLabel(master=self.frame_buttons, text="Appearance Mode:")
        self.label_appearance.grid(row=8, column=0, pady=0, padx=20, sticky="w")

        self.menu_appearance = customtkinter.CTkOptionMenu(master=self.frame_buttons,
                                                           values=["Light", "Dark", "System"],
                                                           command=self.change_appearance_mode)
        self.menu_appearance.grid(row=9, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_button_classify ============

        self.frame_button_classify.rowconfigure(10, weight=10)
        self.frame_button_classify.columnconfigure(0, weight=1)

        self.frame_home = customtkinter.CTkFrame(master=self.frame_button_classify)
        self.frame_home.grid(row=0, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")
        self.frame_home.columnconfigure(1, weight=1)

        # ============ frame_home ============

        self.text_input = customtkinter.CTkTextbox(master=self.frame_home,
                                                   height=130,
                                                   corner_radius=6,
                                                   fg_color=("white", "gray38"))

        self.text_input.grid(column=0, row=0, columnspan=2, sticky="nwe", padx=15, pady=15)

        self.label, self.label_output = [], []

        self.text = ["Configuration " + str(index + 1) + ":\n- " + self.name_classifiers[index]
                     for index in range(len(self.name_classifiers))]

        for index in range(len(self.classifiers)):
            self.label.append(customtkinter.CTkLabel(master=self.frame_home,
                                                     text=self.text[index],
                                                     width=10,
                                                     height=10,
                                                     justify=tkinter.LEFT))
            self.label[index].grid(row=index + 1, column=0, pady=15, padx=15, sticky="nwe")
            self.label_output.append(customtkinter.CTkLabel(master=self.frame_home,
                                                            text="",
                                                            width=300,
                                                            height=80,
                                                            corner_radius=6,
                                                            fg_color=("white", "gray38"),
                                                            justify=tkinter.LEFT))
            self.label_output[index].grid(row=index + 1, column=1, pady=15, padx=15, sticky="nwe")

        # ============ frame_button_classify ============
        self.button_classify = customtkinter.CTkButton(master=self.frame_button_classify,
                                                       text="Classify",
                                                       border_width=2,
                                                       fg_color=None,
                                                       command=self.button_event_classify)
        self.button_classify.grid(row=8, column=1, columnspan=1, pady=20, padx=20, sticky="we")

        # set default values
        self.menu_appearance.set("System")

        self.windows = [None]

    def button_event_classify(self):
        """
        Method to display class prediction and probability of classifiers
        """
        for index in range(len(self.classifiers)):
            self.label_output[index].configure(text=self.classifiers[index].calculate_class(self.text_input.get("0.0", "end")))

    def button_event(self, index):
        """
        Method to open configuration window
        :param index: integer value of index corresponding to the window
        """
        if self.windows[index] is None:
            self.windows[index] = WindowStatistics(self) if index == 0 else None
            self.windows[index].mainloop()

    @staticmethod
    def change_appearance_mode(new_appearance_mode):
        """
        Method to change appearance of the application
        """
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self):
        """
        Method to close all windows of application
        """
        if tkinter.messagebox.askokcancel("Quit", "Do you want to quit?"):
            for window in self.windows:
                try: window.on_closing()
                except (AttributeError, RuntimeError, Exception): pass
            self.destroy()
