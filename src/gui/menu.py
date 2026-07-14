import customtkinter as ctk
import tkinter as tk
from tkinter import ttk

from src.gui.gui_helpers import configure_size, configure_ctk_style
from src.gui.run_window import RunWindow


class MainMenu:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("GenomeML Main Menu")
        self.root.geometry(configure_size(self.root))
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        theme = configure_ctk_style()
        main_frame = ctk.CTkFrame(self.root, fg_color=theme["fg_frame"])  #, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title textboxes
        title = ctk.CTkLabel(
            main_frame, fg_color=theme["fg_label"], 
            text_color=theme["text_label"], 
            text="Welcome to GenomeML", font=theme["font_title"]
        )
        title.grid(row=0, column=0, pady=(0, 20))
        
        subtitle = ctk.CTkLabel(
            main_frame, fg_color=theme["fg_label"], 
            text_color=theme["text_label"], 
            text="(Work in Progress)", font=theme["font_subtitle"]
        )
        subtitle.grid(row=1, column=0, pady=(0, 30))
        
        # Buttons
        btn_run = ctk.CTkButton(
            main_frame, 
            width=180, height=40, 
            bg_color=theme["bg_btn"], fg_color=theme["fg_btn"], 
            text_color=theme["text_btn"], text="Run a Model", 
            font=theme["font_btn"],
            command=self.open_run_window
        )
        btn_run.grid(row=2, column=0, pady=10)
        
        btn_view = ctk.CTkButton(
            main_frame, 
            width=180, height=40, 
            bg_color=theme["bg_btn"], fg_color=theme["fg_btn"], 
            text_color=theme["text_btn"], text="View Outputs", 
            font=theme["font_btn"],
            command=self.view_outputs
        )
        btn_view.grid(row=3, column=0, pady=10)
        
        btn_graph = ctk.CTkButton(
            main_frame, 
            width=180, height=40, 
            bg_color=theme["bg_btn"], fg_color=theme["fg_btn"], 
            text_color=theme["text_btn"], text="Make a Graph", 
            font=theme["font_btn"],
            command=self.make_graph,
        )
        btn_graph.grid(row=4, column=0, pady=10)
        
        btn_exit = ctk.CTkButton(
            main_frame, 
            width=180, height=40, 
            bg_color=theme["bg_btn"], fg_color=theme["fg_btn"], 
            text_color=theme["text_btn"], text="Exit", 
            font=theme["font_btn"],
            command=self.exit_program
        )
        btn_exit.grid(row=5, column=0, pady=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
    
    def open_run_window(self):
        self.root.withdraw()  # Hide main menu
        run_win = RunWindow(self)
    
    def view_outputs(self):
        # Placeholder for future implementation
        pass
    
    def make_graph(self):
        # Placeholder for future implementation
        pass
    
    # Exit the program
    def exit_program(self):
        self.root.quit()
        self.root.destroy()
    
    # Show the main menu window if it is hidden
    def show(self):
        self.root.deiconify()
    
    # Start the GUI main loop
    def run(self):
        self.root.mainloop()

# Entry point for starting the GUI
def start_gui():
    app = MainMenu()
    app.run()