import sys
from PyQt5.QtWidgets import QApplication
from gui.ui.main_window import MainWindow
from gui.utils.style_utils import load_styles


if __name__ == "__main__":
    app = QApplication(sys.argv)
    load_styles(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
