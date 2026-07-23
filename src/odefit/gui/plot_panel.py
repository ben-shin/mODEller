from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QRadioButton, QButtonGroup
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from odefit.plotting.timecourse_plots import plot_dataset_timecourse, plot_variable_comparison
from odefit.plotting.plot_settings import PlotSettings


class PlotPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.main_layout = None
        self.placeholder = None
        self.current_canvas = None

        self.datasets = {}
        self.fit_results = {}  # Memory bank for successful fits!
        self.all_unique_variables = []

        self._setup_ui()

    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)

        # --- Top Section: Mode Selector ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Plot Mode:"))

        self.radio_dataset = QRadioButton("View by Dataset")
        self.radio_variable = QRadioButton("Compare Variables")
        self.radio_dataset.setChecked(True)

        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_dataset)
        self.mode_group.addButton(self.radio_variable)
        self.mode_group.buttonClicked.connect(self._refresh_dropdown)

        mode_layout.addWidget(self.radio_dataset)
        mode_layout.addWidget(self.radio_variable)
        mode_layout.addStretch()
        self.main_layout.addLayout(mode_layout)

        # --- Middle Section: Data Selector ---
        selector_layout = QHBoxLayout()
        self.lbl_selector = QLabel("Select Dataset:")
        selector_layout.addWidget(self.lbl_selector)

        self.combo_selection = QComboBox()
        self.combo_selection.setEnabled(False)
        self.combo_selection.currentTextChanged.connect(self._draw_selected_plot)
        selector_layout.addWidget(self.combo_selection)

        selector_layout.addStretch()
        self.main_layout.addLayout(selector_layout)

        # --- Bottom Section: Plot Area ---
        self.placeholder = QLabel("Load experimental data to see plots...")
        self.placeholder.setStyleSheet("color: gray; font-style: italic; font-size: 14px;")
        self.main_layout.addWidget(self.placeholder)

    def update_datasets(self, datasets_dict):
        """Catches the updated workspace from the Data Panel."""
        self.datasets = datasets_dict

        # Gather every unique variable name across all datasets
        variables = set()
        for ds in self.datasets.values():
            variables.update(ds.signal_columns)
        self.all_unique_variables = sorted(list(variables))

        self._refresh_dropdown()

    def register_fit_result(self, dataset_name, fit_result):
        """Saves a successful fit to memory and forces a redraw if it's currently on screen."""
        self.fit_results[dataset_name] = fit_result
        
        # If the user is currently looking at this dataset, redraw it instantly!
        if self.radio_dataset.isChecked() and self.combo_selection.currentText() == dataset_name:
            self._draw_selected_plot(dataset_name)

    def _refresh_dropdown(self):
        """Updates the dropdown options based on the chosen Mode."""
        self.combo_selection.blockSignals(True)
        self.combo_selection.clear()

        if not self.datasets:
            self.combo_selection.setEnabled(False)
            self._clear_plot()
            self.combo_selection.blockSignals(False)
            return

        self.combo_selection.setEnabled(True)

        if self.radio_dataset.isChecked():
            self.lbl_selector.setText("Select Dataset:")
            self.combo_selection.addItems(list(self.datasets.keys()))
        else:
            self.lbl_selector.setText("Select Variable to Compare:")
            self.combo_selection.addItems(self.all_unique_variables)

        self.combo_selection.blockSignals(False)
        self._draw_selected_plot(self.combo_selection.currentText())

    def _clear_plot(self):
        if self.current_canvas:
            self.main_layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()
            self.current_canvas = None

        if not self.placeholder:
            self.placeholder = QLabel("Load experimental data to see plots...")
            self.placeholder.setStyleSheet("color: gray; font-style: italic; font-size: 14px;")
            self.main_layout.addWidget(self.placeholder)

    def _draw_selected_plot(self, selection):
        """Draws the plot based on the current mode and selection."""
        if not selection:
            return

        if self.radio_dataset.isChecked():
            # MODE 1: Stacked Subplots for a single Dataset
            dataset = self.datasets[selection]
            settings = PlotSettings(
                title=f"Dataset: {selection}",
                x_label=dataset.time_column,
                y_label="Concentration"
            )
            
            # 1. Get the raw figure
            fig, ax_objects = plot_dataset_timecourse(dataset=dataset, settings=settings)
            
            # 2. OVERLAY THE FIT (If we have one saved in memory!)
            if selection in self.fit_results:
                fit_result = self.fit_results[selection]
                
                # Get the current active Matplotlib Axis to draw on
                ax = fig.gca() 
                
                try:
                    # PERFECTLY MATCHED TO optimizer.py
                    timepoints = fit_result.simulation_result.timepoints 
                    
                    for species_name in dataset.signal_columns:
                        # Make sure the simulated species actually exists in the result
                        if species_name in fit_result.simulation_result.species:
                            # Extract the y-values using Ben's SimulationResult method
                            simulated_values = fit_result.simulation_result.get_species_values(species_name)
                            
                            # Draw a thick, smooth line for the mathematical model over the scatter data
                            ax.plot(timepoints, simulated_values, '-', linewidth=2.5, 
                                    label=f"{species_name} (Fit)")
                            
                    ax.legend()
                except AttributeError as e:
                    # Failsafe just in case something goes wrong
                    print(f"Could not draw overlay: {e}")

        else:
            # MODE 2: Overlay a single Variable across all Datasets
            settings = PlotSettings(
                title=f"Comparing '{selection}' across Workspace",
                x_label="Time",
                y_label=selection
            )
            fig, _ = plot_variable_comparison(
                datasets_dict=self.datasets,
                variable=selection,
                settings=settings
            )

        # Display the Figure in the UI
        if self.placeholder:
            self.placeholder.deleteLater()
            self.placeholder = None

        if self.current_canvas:
            self.main_layout.removeWidget(self.current_canvas)
            self.current_canvas.deleteLater()

        self.current_canvas = FigureCanvas(fig)
        self.main_layout.addWidget(self.current_canvas)
