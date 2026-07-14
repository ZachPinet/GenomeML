import customtkinter as ctk
import tkinter as tk
from tkinter import ttk


# This configures the size and position of the window.
def configure_size(window):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    win_width = screen_width // 2
    win_height = screen_height // 2
    x = (screen_width // 2) - (win_width // 2)
    y = (screen_height // 2) - (win_height // 2)

    return f"{win_width}x{win_height}+{x}+{y}"


# This configures the style of the CTK elements.
def configure_ctk_style():
    #ctk.set_appearance_mode("dark")
    #ctk.set_default_color_theme("blue")
    theme = {
        "bg_btn": "#00173c",
        "fg_frame": "#00173c",
        "fg_label": "#00173c",
        "fg_btn": "LightSkyBlue1",
        "font_title": ("Arial", 18, "bold", "underline"),
        "font_subtitle": ("Arial", 10, "italic"),
        "font_btn": ("Arial", 12, "bold"),
        "text_label": "white",
        "text_btn": "black"
    }
    return theme


# This configures the style of the GUI elements.
def configure_style():
    style = ttk.Style()
    style_name = "GenomeML.TFrame"
    style.theme_use('clam')

    style.configure(
        "Title.TLabel", background="#00173c", 
        foreground="white", font=('Arial', 18, 'bold', 'underline')
    )
    style.configure(
        "Subtitle.TLabel", background="#00173c", 
        foreground="white", font=('Arial', 10, 'italic')
    )
    style.configure(
        "Button.TButton", background="lightgray", 
        foreground="black", font=('Arial', 12, 'bold')
    )
    style.configure(
        "ButtonFrame.TFrame", background="#00173c"
    )
    style.configure(style_name, background='#00173c')

    return style_name


# This handles mouse wheel scrolling for canvas movement.
def on_mousewheel(canvas, event):
    current_top = canvas.canvasy(0)
    scroll_amount = int(-1*(event.delta/120))

    if scroll_amount < 0 and current_top <= 0:
        return
    
    canvas.yview_scroll(scroll_amount, "units")