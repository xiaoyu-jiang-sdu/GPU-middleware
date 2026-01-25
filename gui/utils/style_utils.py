from PyQt5.QtWidgets import QApplication

from config.project_config import ProjectConfig


def load_styles(app: QApplication):
    base_dir = ProjectConfig.gui_dir()
    styles_dir = base_dir / "styles"

    qss = ""

    for path in sorted(styles_dir.glob("*.qss")):
        try:
            qss += path.read_text(encoding="utf-8") + "\n"
        except Exception as e:
            print(f"[StyleLoader] Failed to load {path.name}: {e}")

    app.setStyleSheet(qss)
