import os, csv, sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView
)
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt, QTimer

CSV_PATH = os.path.join(os.path.dirname(__file__), "solution.csv")

class SolutionEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("5×5 Solution Editor")
        self.resize(600, 600)

        self.table = QTableWidget(5, 5)
        self.table.setEditTriggers(QTableWidget.DoubleClicked)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setHorizontalHeaderLabels(["A", "B", "C", "D", "E"])

        # ← NEW: timer + pending cell for click vs double‑click
        self._pending = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._toggle_pending)
        self.table.cellDoubleClicked.connect(self._cancel_pending)

        # Connect the delayed toggle instead of direct toggle
        self.table.cellClicked.connect(self._schedule_toggle)
        self.table.itemChanged.connect(self.save_csv)

        central = QWidget()
        central.setLayout(QVBoxLayout())
        central.layout().addWidget(self.table)
        self.setCentralWidget(central)

        self.load_csv()

    def load_csv(self):
        grid = [["" for _ in range(5)] for __ in range(5)]
        if os.path.exists(CSV_PATH):
            with open(CSV_PATH, newline="", encoding="utf-8") as f:
                for i, row in enumerate(csv.reader(f)):
                    for j, raw in enumerate(row[:5]):
                        grid[i][j] = raw.strip()

        for i in range(5):
            for j in range(5):
                text = grid[i][j]
                filled = text.startswith("#")
                label = text.lstrip("#").strip()
                item = QTableWidgetItem(label)
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QBrush(QColor("black") if filled else QColor("white")))
                item.setForeground(QBrush(QColor("white") if filled else QColor("black")))
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.table.setItem(i, j, item)

    # ← NEW: schedule toggle after single‑click
    def _schedule_toggle(self, row, col):
        self._pending = (row, col)
        self._timer.start(QApplication.instance().doubleClickInterval())

    # ← NEW: cancel toggle on double‑click
    def _cancel_pending(self, row, col):
        self._timer.stop()
        self._pending = None

    # ← NEW: actual toggle logic, called only if no double‑click happened
    def _toggle_pending(self):
        if not self._pending:
            return
        row, col = self._pending
        item = self.table.item(row, col)
        bg = item.background().color().name()
        filled = (bg != "#000000")
        item.setBackground(QBrush(QColor("black") if filled else QColor("white")))
        item.setForeground(QBrush(QColor("white") if filled else QColor("black")))
        self.save_csv()
        self._pending = None

    def save_csv(self, *args):
        rows = []
        for i in range(5):
            row = []
            for j in range(5):
                item = self.table.item(i, j)
                if item is None:
                    row.append("")          # empty cell
                    continue

                txt = item.text().strip()
                filled = item.background().color().name() == "#000000"
                row.append(f"#{txt}" if filled and txt else txt)
            rows.append(row)

        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = SolutionEditor()
    editor.show()
    sys.exit(app.exec_())
