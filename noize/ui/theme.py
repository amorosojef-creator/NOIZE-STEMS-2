"""NOIZE design system: flat, 1px borders, max 3px radius, no gradients or shadows."""
BG = "#101214"
BG2 = "#181B1F"
PANEL = "#202428"
LINE = "#353B40"
ACCENT = "#CCFF00"
TEXT = "#F2F4F1"
TEXT2 = "#929AA0"
RADIUS = 3

STYLE = f"""
* {{ font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif; font-size: 12px; color: {TEXT}; }}
QMainWindow, QDialog, QWidget#root {{ background: {BG}; }}
QWidget {{ background: transparent; }}
QWidget#panel {{ background: {BG2}; border: 1px solid {LINE}; }}
QWidget#topbar {{ background: {BG2}; border-bottom: 1px solid {LINE}; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: {RADIUS}px; }}
QLabel#h {{ color: {TEXT2}; font-size: 10px; letter-spacing: 2px; font-weight: 600; }}
QLabel#brand {{ font-size: 15px; font-weight: 700; letter-spacing: 4px; }}
QLabel#dim {{ color: {TEXT2}; }}
QLabel#title {{ font-size: 15px; font-weight: 600; }}
QPushButton {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: {RADIUS}px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BG}; }}
QPushButton:disabled {{ color: #5b6268; border-color: #2a2f33; }}
QPushButton#primary {{ background: {ACCENT}; color: {BG}; border: 1px solid {ACCENT}; font-weight: 700; letter-spacing: 1px; }}
QPushButton#primary:disabled {{ background: #3b4400; color: #7b8433; border-color: #3b4400; }}
QPushButton#primary:hover {{ background: #DBFF4D; }}
QPushButton#nav {{ background: transparent; border: none; border-bottom: 2px solid transparent; border-radius: 0;
    padding: 14px 16px; letter-spacing: 2px; font-weight: 600; color: {TEXT2}; }}
QPushButton#nav:hover {{ color: {TEXT}; }}
QPushButton#nav:checked {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QPushButton#sq {{ padding: 4px 0; }}
QPushButton#tog {{ padding: 2px 6px; min-width: 14px; font-weight: 700; }}
QPushButton#tog:checked {{ background: {ACCENT}; color: {BG}; border-color: {ACCENT}; }}
QPushButton#tog:checked {{ color: {BG}; }}
QComboBox, QLineEdit, QSpinBox {{ background: {BG}; border: 1px solid {LINE}; border-radius: {RADIUS}px; padding: 4px 8px; min-height: 18px; }}
QComboBox:hover, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {BG2}; border: 1px solid {LINE}; selection-background-color: {ACCENT}; selection-color: {BG}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QListWidget, QTableWidget {{ background: {BG}; border: 1px solid {LINE}; gridline-color: {LINE}; alternate-background-color: {BG2}; }}
QListWidget::item {{ padding: 5px 6px; }}
QListWidget::item:selected, QTableWidget::item:selected {{ background: {PANEL}; color: {ACCENT}; }}
QHeaderView::section {{ background: {BG2}; color: {TEXT2}; border: none; border-bottom: 1px solid {LINE}; padding: 5px; font-size: 10px; letter-spacing: 1px; }}
QProgressBar {{ background: {BG}; border: 1px solid {LINE}; border-radius: {RADIUS}px; height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::groove:horizontal {{ height: 2px; background: {LINE}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 8px; height: 12px; margin: -5px 0; border-radius: 1px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 12px; height: 12px; border: 1px solid {LINE}; background: {BG}; border-radius: 2px; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QScrollBar:vertical {{ background: {BG}; width: 8px; }}
QScrollBar::handle:vertical {{ background: {LINE}; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: {BG}; height: 8px; }}
QScrollBar::handle:horizontal {{ background: {LINE}; min-width: 24px; }}
QToolTip {{ background: {PANEL}; color: {TEXT}; border: 1px solid {LINE}; }}
QMessageBox {{ background: {BG2}; }}
"""
