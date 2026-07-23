import pandas as pd
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QListWidgetItem,
    QMessageBox, QAbstractItemView, QGroupBox
)
from PySide6.QtCore import Signal

from odefit.data.csv_reader import read_csv_dataset


class DataImportPanel(QWidget):
    datasets_updated = Signal(dict)

    def __init__(self):
        super().__init__()
        self.current_filepath = None
        self.loaded_datasets = {}  
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Top Section: File Loading ---
        import_group = QGroupBox("1. Import New Dataset")
        import_layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        self.btn_browse = QPushButton("Browse CSV File...")
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        self.lbl_filepath = QLabel("No file selected.")
        self.lbl_filepath.setStyleSheet("color: gray; font-style: italic;")
        file_layout.addWidget(self.btn_browse)
        file_layout.addWidget(self.lbl_filepath, stretch=1)
        import_layout.addLayout(file_layout)

        mapping_layout = QHBoxLayout()
        
        # Replaced brittle QComboBox with a bulletproof QListWidget
        time_layout = QVBoxLayout()
        time_layout.addWidget(QLabel("Time Column:"))
        self.list_time = QListWidget()
        self.list_time.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        time_layout.addWidget(self.list_time)
        mapping_layout.addLayout(time_layout)

        signal_layout = QVBoxLayout()
        signal_layout.addWidget(QLabel("Variable Column(s):"))
        self.list_signals = QListWidget()
        self.list_signals.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        signal_layout.addWidget(self.list_signals)
        mapping_layout.addLayout(signal_layout)

        import_layout.addLayout(mapping_layout)

        self.btn_load = QPushButton("Add Dataset to Workspace")
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_load.setEnabled(False)
        import_layout.addWidget(self.btn_load)

        import_group.setLayout(import_layout)
        main_layout.addWidget(import_group)

        # --- Bottom Section: Loaded Workspace ---
        workspace_group = QGroupBox("2. Loaded Workspace")
        workspace_layout = QVBoxLayout()

        self.list_loaded_datasets = QListWidget()
        workspace_layout.addWidget(self.list_loaded_datasets)

        btn_clear = QPushButton("Clear Workspace")
        btn_clear.clicked.connect(self._on_clear_clicked)
        workspace_layout.addWidget(btn_clear)

        workspace_group.setLayout(workspace_layout)
        main_layout.addWidget(workspace_group)

    def _on_browse_clicked(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Experimental Data", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filepath:
            self.current_filepath = filepath
            self.lbl_filepath.setText(filepath)
            self._populate_columns(filepath)

    def _populate_columns(self, filepath: str):
        try:
            df_preview = pd.read_csv(filepath_or_buffer=filepath, nrows=0)
            assert isinstance(df_preview, pd.DataFrame)
            columns = df_preview.columns.tolist()

            self.list_time.clear()
            self.list_signals.clear()
            
            for col in columns:
                self.list_time.addItem(QListWidgetItem(col))
                self.list_signals.addItem(QListWidgetItem(col))

            self.btn_load.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{str(e)}")

    def _on_load_clicked(self):
        time_items = self.list_time.selectedItems()
        selected_items = self.list_signals.selectedItems()

        if not time_items:
            QMessageBox.warning(self, "Warning", "Please select exactly one Time column.")
            return

        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one Variable column.")
            return

        time_col = time_items[0].text()
        signal_cols = [item.text() for item in selected_items]

        try:
            dataset = read_csv_dataset(self.current_filepath, time_col, signal_cols)
            dataset_name = Path(self.current_filepath).name

            self.loaded_datasets[dataset_name] = dataset

            self.list_loaded_datasets.clear()
            self.list_loaded_datasets.addItems(list(self.loaded_datasets.keys()))

            self.datasets_updated.emit(self.loaded_datasets)
            QMessageBox.information(self, "Success", f"Added '{dataset_name}' to workspace!")

        except Exception as e:
            QMessageBox.critical(self, "Validation Error", str(e))

    def _on_clear_clicked(self):
        self.loaded_datasets.clear()
        self.list_loaded_datasets.clear()
        self.datasets_updated.emit(self.loaded_datasets)
