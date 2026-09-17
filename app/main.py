"""Application entry point: `python -m app.main` (or ./run_app.sh)."""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from app.config import APP_NAME, AppConfig
    from app.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("HoloSmart")

    config = AppConfig.load()
    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
