import threading

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from src import config
from src.gui.gui_helpers import configure_size, configure_style, on_mousewheel
from src.training_runner import run_training


class RunWindow:
    # Class variable shared across all instances.
    # This lets a user close the window and keep the button's status.
    _training_in_progress = False
    
    # Set up window variables and initialize the GUI
    def __init__(self, parent_menu):
        self.parent_menu = parent_menu
        self.root = tk.Toplevel()
        self.root.title("Configure Run Variables")
        self.root.geometry(configure_size(self.root))
        
        # Variables for workflow and file selection data
        self.start_file_var = tk.IntVar(value=1)
        self.end_file_var = tk.IntVar(value=1)
        self.file_mode_var = tk.StringVar(value="single")
        self.workflow_var = tk.StringVar(value="single_column")

        # Workflow-specific options
        self.kfold_var = tk.BooleanVar()
        self.pca_components_var = tk.IntVar()
        self.data_splits_var = tk.IntVar()
        self.double_train_file_var = tk.IntVar()
        
        # Additional options
        self.max_seqs_var = tk.IntVar()
        self.train_percentage_var = tk.IntVar()
        self.verbose_var = tk.IntVar()
        self.make_plot_var = tk.BooleanVar()
        self.show_bounds_var = tk.BooleanVar()
        self.std_multiplier_var = tk.DoubleVar()
        self.frac_var = tk.DoubleVar()
        self.outlier_mode_var = tk.StringVar()
        self.random_seed_var = tk.IntVar()
        self.window_var = tk.BooleanVar()
        self.use_batch_correction_var = tk.BooleanVar()
        self.complex_outlier_file_var = tk.StringVar()
        self.percent_outlier_file_var = tk.StringVar()
        self.run_data_file_var = tk.StringVar()
        
        # Initialize the GUI
        self.load_current_config()
        self.create_widgets()
        self.update_workflow_options()
        self.update_file_options()
        
        # Prevent spinboxes from auto-focusing and capturing mousewheel
        self.root.after(100, self._set_initial_focus)
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Check if training is already in progress when window opens
        if RunWindow._training_in_progress:
            # Delay the button state update until after widgets are created
            self.root.after(100, self._update_run_button)

    # Force an update of the run button state if training is in progess.
    def _update_run_button(self):
        if RunWindow._training_in_progress:
            self.btn_run.config(state='disabled', text="Training...")
    
    # Set initial focus to prevent spinbox auto-focus
    def _set_initial_focus(self):
        # Focus on the window itself instead of any widget
        self.root.focus_force()
    
    # Disable mousewheel on spinbox to prevent value changes while scrolling
    def _disable_spinbox_mousewheel(self, spinbox):
        def ignore_mousewheel(event):
            # Remove focus from spinbox when mousewheel is used
            self.root.focus()
            return "break"  # Prevent event from propagating
        
        spinbox.bind("<MouseWheel>", ignore_mousewheel)

    # Get the local config values and load them into the GUI variables
    def load_current_config(self):
        # Set file selections based on mode
        if config.DO_RANGE:
            self.start_file_var.set(config.RANGE_START_FILE_NUM)
            self.end_file_var.set(config.RANGE_END_FILE_NUM)
        else:
            self.start_file_var.set(config.SINGLE_FILE_NUM)
            self.end_file_var.set(config.SINGLE_FILE_NUM)

        # Set file mode
        self.file_mode_var.set("range" if config.DO_RANGE else "single")

        # Set workflow based on current local config values
        if config.DO_SINGLE_COLUMN:
            self.workflow_var.set("single_column")
        elif config.DO_PCA:
            self.workflow_var.set("pca")
        elif config.DO_ENSEMBLE:
            self.workflow_var.set("ensemble")
        elif config.DO_DOUBLE_COLUMNS:
            self.workflow_var.set("double_columns")
        
        # Workflow options
        self.kfold_var.set(config.KFOLD)
        self.pca_components_var.set(config.PCA_COMPONENTS)
        self.data_splits_var.set(config.DATA_SPLITS)
        self.double_train_file_var.set(config.DOUBLE_TRAIN_FILE)
        
        # Additional options
        self.max_seqs_var.set(config.MAX_SEQS)
        self.train_percentage_var.set(config.TRAIN_PERCENTAGE)
        self.verbose_var.set(config.VERBOSE)
        self.make_plot_var.set(config.MAKE_PLOT)
        self.show_bounds_var.set(config.SHOW_BOUNDS)
        self.std_multiplier_var.set(config.STD_MULTIPLIER)
        self.frac_var.set(config.FRAC)
        self.outlier_mode_var.set(config.OUTLIER_MODE)
        self.random_seed_var.set(config.RANDOM_SEED)
        self.window_var.set(config.WINDOW)
        self.use_batch_correction_var.set(config.USE_BATCH_CORRECTION)
        self.complex_outlier_file_var.set(config.COMPLEX_OUTLIER_FILE)
        self.percent_outlier_file_var.set(config.PERCENT_OUTLIER_FILE)
        self.run_data_file_var.set(config.RUN_DATA_FILE)
    
    # Save current GUI values to config_local.txt
    def save_config_to_file(self):
        config_file = Path(__file__).parent.parent / 'config_local.txt'
        
        # Determine workflow flags
        workflow = self.workflow_var.get()
        do_single = workflow == "single_column"
        do_pca = workflow == "pca"
        do_ensemble = workflow == "ensemble"
        do_double = workflow == "double_columns"
        
        # Determine file mode
        do_range = self.file_mode_var.get() == "range"
        
        # Get file numbers - store as 1-indexed for user-friendliness
        single_file_num = self.start_file_var.get() if not do_range else config.SINGLE_FILE_NUM
        range_start = self.start_file_var.get() if do_range else config.RANGE_START_FILE_NUM
        range_end = self.end_file_var.get() if do_range else config.RANGE_END_FILE_NUM
        double_train_file = self.double_train_file_var.get()
        
        config_lines = [
            "# GenomeML Configuration Overrides",
            f"SINGLE_FILE_NUM={single_file_num}",
            f"RANGE_START_FILE_NUM={range_start}",
            f"RANGE_END_FILE_NUM={range_end}",
            f"DO_RANGE={do_range}",
            f"DO_DOUBLE_COLUMNS={do_double}",
            f"DO_ENSEMBLE={do_ensemble}",
            f"DO_PCA={do_pca}",
            f"DO_SINGLE_COLUMN={do_single}",
            f"KFOLD={self.kfold_var.get()}",
            f"PCA_COMPONENTS={self.pca_components_var.get()}",
            f"DATA_SPLITS={self.data_splits_var.get()}",
            f"DOUBLE_TRAIN_FILE={double_train_file}",
            f"MAX_SEQS={self.max_seqs_var.get()}",
            f"TRAIN_PERCENTAGE={self.train_percentage_var.get()}",
            f"VERBOSE={self.verbose_var.get()}",
            f"MAKE_PLOT={self.make_plot_var.get()}",
            f"SHOW_BOUNDS={self.show_bounds_var.get()}",
            f"STD_MULTIPLIER={self.std_multiplier_var.get()}",
            f"FRAC={self.frac_var.get()}",
            f"OUTLIER_MODE={self.outlier_mode_var.get()}",
            f"RANDOM_SEED={self.random_seed_var.get()}",
            f"WINDOW={self.window_var.get()}",
            f"USE_BATCH_CORRECTION={self.use_batch_correction_var.get()}",
            f"COMPLEX_OUTLIER_FILE={self.complex_outlier_file_var.get()}",
            f"PERCENT_OUTLIER_FILE={self.percent_outlier_file_var.get()}",
            f"RUN_DATA_FILE={self.run_data_file_var.get()}",
        ]
        
        # Edit the local config file
        try:
            with open(config_file, 'w') as f:
                f.write('\n'.join(config_lines))
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    # Create the frames and widgets for the GUI
    def create_widgets(self):
        # Main frame with scrollbar
        style_name = configure_style()

        # Create main frame (container for scrollbar)
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        # Canvas for scrolling
        canvas = tk.Canvas(container, highlightthickness=0, bg='#00173c')
        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=canvas.yview
        )

        # Main content frame
        main_frame = ttk.Frame(canvas, style=style_name, padding="20")

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)

        # Configure scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_frame_id = canvas.create_window(
            (0, 0), window=main_frame, anchor="n"
        )

        # And add this function to center the frame:
        def center_frame_in_canvas(event=None):
            canvas.update_idletasks()  # Ensure canvas is properly sized
            canvas_width = canvas.winfo_width()
            if canvas_width > 1:
                # Center the frame horizontally in the canvas
                canvas.coords(canvas_frame_id, canvas_width // 2, 0)
                canvas.itemconfig(canvas_frame_id, width=canvas_width)

        # Update scroll region when content changes
        def update_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            center_frame_in_canvas()
        
        main_frame.bind('<Configure>', update_scroll_region)
        canvas.bind('<Configure>', center_frame_in_canvas)
        
        # Mouse wheel scrolling support        
        canvas.bind(
            "<MouseWheel>", lambda event: on_mousewheel(canvas, event)
        )
        self.root.bind(
            "<MouseWheel>", lambda event: on_mousewheel(canvas, event)
        )

        # Configure the main_frame grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.columnconfigure(2, weight=1)
        
        # Title
        title = ttk.Label(
            main_frame, text="Configure Run Variables", style="Title.TLabel"
        )
        title.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Workflow selection frame
        self.workflow_frame = ttk.LabelFrame(
            main_frame, text="   Workflow Mode", padding="5"
        )
        self.workflow_frame.grid(
            row=1, column=0, pady=10, padx=(0, 10), sticky="E"
        )
        self.workflow_frame.configure(width=200, height=120)
        self.workflow_frame.grid_propagate(False)
        
        # File selection frame
        self.file_frame = ttk.LabelFrame(
            main_frame, text="   File Selection", padding="5"
        )
        self.file_frame.grid(
            row=1, column=2, pady=10, padx=(10, 0), sticky="W"
        )
        self.file_frame.configure(width=200, height=120)
        self.file_frame.grid_propagate(False)
        
        # Workflow options frame
        self.options_frame = ttk.LabelFrame(
            main_frame, text="   Workflow Options", padding="5"
        )
        self.options_frame.grid(
            row=2, column=0, pady=10, padx=(0, 10), sticky="E"
        )
        self.options_frame.configure(width=200, height=120)
        self.options_frame.grid_propagate(False)

        # File options frame
        self.file_options_frame = ttk.LabelFrame(
            main_frame, text="   File Options", padding="5"
        )
        self.file_options_frame.grid(
            row=2, column=2, pady=10, padx=(10, 0), sticky="W"
        )
        self.file_options_frame.configure(width=200, height=120)
        self.file_options_frame.grid_propagate(False)

        # Additional options frame
        self.additional_options_frame = ttk.LabelFrame(
            main_frame, text="   Additional Options", padding="10"
        )
        self.additional_options_frame.grid(
            row=3, column=0, columnspan=3, pady=10
        )
        self.additional_options_frame.configure(width=420, height=120)
        self.additional_options_frame.grid_propagate(False)
        
        # Button frame (Start Training and Back to Menu)
        self.button_frame = ttk.Frame(
            main_frame, style="ButtonFrame.TFrame", padding="10"
        )
        self.button_frame.grid(row=4, column=0, columnspan=3, pady=10)

        # Populate the frames with options based on the config variables
        self.populate_options()

    # Populate the frames with options based on the config variables
    def populate_options(self):
        # Workflow selection fields
        ttk.Radiobutton(
            self.workflow_frame, text="Single Column", 
            variable=self.workflow_var, value="single_column",
            command=self.update_workflow_options
        ).grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Radiobutton(
            self.workflow_frame, text="PCA", 
            variable=self.workflow_var, value="pca",
            command=self.update_workflow_options
        ).grid(
            row=1, column=0, sticky=tk.W
        )
        ttk.Radiobutton(
            self.workflow_frame, text="Ensemble", 
            variable=self.workflow_var, value="ensemble",
            command=self.update_workflow_options
        ).grid(
            row=2, column=0, sticky=tk.W
        )
        ttk.Radiobutton(
            self.workflow_frame, text="Double Columns", 
            variable=self.workflow_var, value="double_columns",
            command=self.update_workflow_options
        ).grid(
            row=3, column=0, sticky=tk.W
        )

        # File selection fields
        ttk.Radiobutton(
            self.file_frame, text="Single File", 
            variable=self.file_mode_var, value="single",
            command=self.update_file_options
        ).grid(
            row=0, column=0, sticky=tk.W
        )
        ttk.Radiobutton(
            self.file_frame, text="Range of Files", 
            variable=self.file_mode_var, value="range",
            command=self.update_file_options
        ).grid(
            row=1, column=0, sticky=tk.W
        )

        # Additional options row 1
        ttk.Label(
            self.additional_options_frame, text="Max Sequences:"
        ).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 5)
        )
        max_seqs_spin = ttk.Spinbox(
            self.additional_options_frame,
            from_=1000, to=999999, width=10,
            textvariable=self.max_seqs_var
        )
        max_seqs_spin.grid(
            row=1, column=0, sticky=tk.W, padx=(0, 20)
        )
        self._disable_spinbox_mousewheel(max_seqs_spin)
        
        ttk.Label(
            self.additional_options_frame, text="Train Percentage:"
        ).grid(
            row=0, column=1, sticky=tk.W, padx=(0, 5)
        )
        train_pct_spin = ttk.Spinbox(
            self.additional_options_frame,
            from_=10, to=90, width=10,
            textvariable=self.train_percentage_var
        )
        train_pct_spin.grid(
            row=1, column=1, sticky=tk.W, padx=(0, 20)
        )
        self._disable_spinbox_mousewheel(train_pct_spin)
        
        ttk.Label(
            self.additional_options_frame, text="Random Seed:"
        ).grid(
            row=0, column=2, sticky=tk.W, padx=(0, 5)
        )
        random_seed_spin = ttk.Spinbox(
            self.additional_options_frame,
            from_=1, to=9999, width=10,
            textvariable=self.random_seed_var
        )
        random_seed_spin.grid(
            row=1, column=2, sticky=tk.W
        )
        self._disable_spinbox_mousewheel(random_seed_spin)
        
        # Additional options row 2
        ttk.Checkbutton(
            self.additional_options_frame,
            text="Make Plots",
            variable=self.make_plot_var
        ).grid(
            row=2, column=0, sticky=tk.W, pady=(10, 0)
        )
        
        ttk.Checkbutton(
            self.additional_options_frame,
            text="Show Bounds",
            variable=self.show_bounds_var
        ).grid(
            row=2, column=1, sticky=tk.W, pady=(10, 0)
        )
        
        ttk.Label(
            self.additional_options_frame, text="Outlier Mode:"
        ).grid(
            row=2, column=2, sticky=tk.W, pady=(10, 0)
        )
        ttk.Combobox(
            self.additional_options_frame,
            textvariable=self.outlier_mode_var,
            values=['simple', 'complex', 'both', 'off'],
            state="readonly",
            width=8
        ).grid(
            row=2, column=3, sticky=tk.W, pady=(10, 0)
        )

        # Run and Back buttons
        self.btn_run = ttk.Button(
            self.button_frame, text="Run Model", style="Button.TButton", command=self.start_training, width=15
        )
        self.btn_run.grid(row=0, column=0, padx=10)
        self.btn_back = ttk.Button(
            self.button_frame, text="Back to Menu", style="Button.TButton", command=self.back_to_menu, width=15
        )
        self.btn_back.grid(row=0, column=1, padx=10)
        
    # Get list of .txt files in inputs/columns/ with 1-based indexing
    def get_column_files(self):
        columns_dir = Path.cwd() / 'inputs' / 'columns'
        if not columns_dir.exists():
            return []
        
        # Add file's number to the display name
        file_options = []
        txt_files = sorted(columns_dir.glob("*.txt"))
        for i, file_path in enumerate(txt_files, 1):
            display_name = f"{i} - {file_path.stem}"
            file_options.append(display_name)
        
        return file_options
    
    # Update the options frame based on selected workflow
    def update_workflow_options(self):
        # Clear existing widgets
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        
        workflow = self.workflow_var.get()
        
        # K-Fold option for single column workflow
        if workflow == "single_column":
            s_button = ttk.Checkbutton(
                self.options_frame, 
                text="Use K-Fold Cross Validation",
                variable=self.kfold_var
            )
            s_button.grid(row=0, column=0, sticky=tk.W)
        
        # PCA components for PCA workflow
        elif workflow == "pca":
            ttk.Label(
                self.options_frame, text="PCA Components:"
            ).grid(
                row=0, column=0, sticky=tk.W, padx=(0, 10)
            )
            pca_spin = ttk.Spinbox(
                self.options_frame, 
                from_=1, to=20, width=5, 
                textvariable=self.pca_components_var
            )
            pca_spin.grid(
                row=0, column=1, sticky=tk.W
            )
            self._disable_spinbox_mousewheel(pca_spin)
        
        # Data splits for ensemble workflow
        elif workflow == "ensemble":            
            ttk.Label(
                self.options_frame, text="Data Splits:"
            ).grid(
                row=0, column=0, sticky=tk.W, padx=(0, 10)
            )
            data_splits_spin = ttk.Spinbox(
                self.options_frame, 
                from_=2, to=20, width=5, 
                textvariable=self.data_splits_var
            )
            data_splits_spin.grid(
                row=0, column=1, sticky=tk.W
            )
            self._disable_spinbox_mousewheel(data_splits_spin)
        
        # Base file for double columns workflow
        elif workflow == "double_columns":
            ttk.Label(
                self.options_frame, text=(
                    "Choose the file to train on:\n"
                    "(File Selection files are tested on)"
                )
            ).grid(
                row=0, column=0, sticky=tk.W
            )

            file_options = self.get_column_files()
            if file_options:
                file_dropdown = ttk.Combobox(
                    self.options_frame, 
                    values=file_options, 
                    state="readonly", 
                    width=25,
                )
                file_dropdown.grid(row=1, column=0, sticky=tk.W)

                # Set current value if valid
                current_val = self.double_train_file_var.get()
                if 1 <= current_val <= len(file_options):
                    file_dropdown.set(file_options[current_val - 1])
                if file_options:
                    file_dropdown.set(file_options[0])
                    self.double_train_file_var.set(1)

                # Bind the selection changes
                file_dropdown.bind(
                    '<<ComboboxSelected>>', 
                    lambda e: self.on_file_select(
                        e, target_var=self.double_train_file_var
                    ), 
                )
            
            else:
                ttk.Label(
                    self.options_frame, text="No .txt files found"
                ).grid(
                    row=1, column=0, sticky=tk.W
                )

    # Update the file options frame based on selected file mode
    def update_file_options(self):
        # Clear existing widgets
        for widget in self.file_options_frame.winfo_children():
            widget.destroy()
        
        file_mode = self.file_mode_var.get()
        
        if file_mode == "single":
            ttk.Label(
                self.file_options_frame, text="Select a file to run on:"
            ).grid(
                row=0, column=0, pady=(0, 5), sticky=tk.W
            )
            
            file_options = self.get_column_files()
            if file_options:
                file_dropdown = ttk.Combobox(
                    self.file_options_frame, 
                    values=file_options, 
                    state="readonly", 
                    width=25
                )
                file_dropdown.grid(row=1, column=0, sticky=tk.W)

                # Restore from config values
                single_file = config.SINGLE_FILE_NUM
                if 1 <= single_file <= len(file_options):
                    file_dropdown.set(file_options[single_file - 1])
                    self.start_file_var.set(single_file)
                    self.end_file_var.set(single_file)
                elif file_options:
                    file_dropdown.set(file_options[0])
                    self.start_file_var.set(1)
                    self.end_file_var.set(1)

                file_dropdown.bind('<<ComboboxSelected>>', self.on_file_select)

            else:
                ttk.Label(
                    self.file_options_frame, text="No .txt files found"
                ).grid(
                    row=1, column=0, sticky=tk.W
                )
        
        elif file_mode == "range":
            ttk.Label(
                self.file_options_frame, text="Select start and end files:"
            ).grid(
                row=0, column=0, columnspan=2, pady=(0, 5), sticky=tk.W
            )
            
            max_files = len(self.get_column_files())
            
            # Restore from config values
            range_start = config.RANGE_START_FILE_NUM
            range_end = config.RANGE_END_FILE_NUM
            if 1 <= range_start <= max_files:
                self.start_file_var.set(range_start)
            else:
                self.start_file_var.set(1)
            if 1 <= range_end <= max_files:
                self.end_file_var.set(range_end)
            else:
                self.end_file_var.set(1)
            
            # Start of the range
            ttk.Label(
                self.file_options_frame, text="Start:"
            ).grid(
                row=1, column=0, sticky=tk.W, padx=(0, 5)
            )
            start_spin = ttk.Spinbox(
                self.file_options_frame, 
                from_=1, to=max_files, 
                width=5, 
                textvariable=self.start_file_var
            )
            start_spin.grid(row=1, column=1, sticky=tk.W)
            self._disable_spinbox_mousewheel(start_spin)
            
            # End of the range
            ttk.Label(
                self.file_options_frame, text="End:"
            ).grid(
                row=2, column=0, sticky=tk.W, padx=(0, 5)
            )
            end_spin = ttk.Spinbox(
                self.file_options_frame, 
                from_=1, to=max_files,
                width=5,
                textvariable=self.end_file_var
            )
            end_spin.grid(row=2, column=1, sticky=tk.W)
            self._disable_spinbox_mousewheel(end_spin)

    # Handle file selection from the dropdown
    def on_file_select(self, event, target_var=None):
        selected_file = event.widget.get()

        # Default fallback for invalid selection
        if not selected_file or " - " not in selected_file:
            if target_var:
                target_var.set(1)
            else:
                self.start_file_var.set(1)
                self.end_file_var.set(1)
            return
        
        # Extract number from "1 - filename" format
        file_num = int(selected_file.split(' - ')[0])

        # Double column train file or regular file selection
        if target_var is not None:
            target_var.set(file_num)
        else:
            self.start_file_var.set(file_num)
            self.end_file_var.set(file_num)
    
    # Save selected options and start the training process
    def start_training(self):
        self.save_config_to_file()
        config.reload_config()

        # Set class variable to indicate training started
        RunWindow._training_in_progress = True

        # Disable the Run button during training
        self.btn_run.config(state='disabled', text="Training...")

        # Define the background training function inline
        def run_training_background():
            try:
                print(f"Starting {self.workflow_var.get()} training...")
                run_training()
            except Exception as e:
                print(f"Training error: {e}")
            finally:
                # Reset class variable when training is done
                RunWindow._training_in_progress = False

                # Only try to update GUI if the window is still open
                try:
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        self.root.after(
                            0, lambda: self.btn_run.config(
                                state='normal', text="Start Training"
                            )
                        )
                # Continue if GUI was closed
                except (tk.TclError, RuntimeError):
                    pass

        # Run training in background thread
        training_thread = threading.Thread(target=run_training_background)
        training_thread.daemon = False
        training_thread.start()

        print("Training started in background thread...")
    
    # Return to the main menu and close this window
    def back_to_menu(self):
        print("Window closing by return to main menu.")
        self.root.destroy()
        self.parent_menu.show()
    
    # Handle window closing
    def on_closing(self):
        print("Window closed by X button.")
        self.back_to_menu()