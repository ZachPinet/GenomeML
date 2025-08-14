import tkinter as tk
from tkinter import ttk

from .gui_helpers import configure_size, configure_style
from .run_window import RunWindow


class MainMenu:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GenomeML Main Menu")
        self.root.geometry(configure_size(self.root))
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        style_name = configure_style()
        main_frame = ttk.Frame(self.root, style=style_name, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title textboxes
        title = ttk.Label(
            main_frame, text="Welcome to GenomeML", style="Title.TLabel"
        )
        title.grid(row=0, column=0, pady=(0, 20))
        
        subtitle = ttk.Label(
            main_frame, text="(Work in Progress)", style="Subtitle.TLabel"
        )
        subtitle.grid(row=1, column=0, pady=(0, 30))
        
        # Buttons
        btn_run = ttk.Button(
            main_frame, text="Run a Model", style="Button.TButton",
            command=self.open_run_window, width=20
        )
        btn_run.grid(row=2, column=0, pady=10)
        
        btn_view = ttk.Button(
            main_frame, text="View Outputs", style="Button.TButton",
            command=self.view_outputs, width=20
        )
        btn_view.grid(row=3, column=0, pady=10)
        
        btn_graph = ttk.Button(
            main_frame, text="Make a Graph", style="Button.TButton",
            command=self.make_graph, width=20
        )
        btn_graph.grid(row=4, column=0, pady=10)
        
        btn_exit = ttk.Button(
            main_frame, text="Exit", style="Button.TButton",
            command=self.exit_program, width=20
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