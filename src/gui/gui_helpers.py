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