from src import config
from src.gui.menu import start_gui
from src.training_runner import run_training


# This is the program's main function.
def main():
    # Check if GUI mode is enabled
    if config.WINDOW:
        start_gui()
    else:
        run_training()

    print("Program is finished.")