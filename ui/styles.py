# Color palette
BG_DEEP = "#23103f"
BG_CARD = "#352059"
BG_CARD_HOVER = "#422a69"
ACCENT_PRIMARY = "#7c3aed"
ACCENT_GLOW = "#a855f7"
ACCENT_SUCCESS = "#b879ff"
TEXT_PRIMARY = "#f0e6ff"
TEXT_SECONDARY = "#baa2e9"
TEXT_DISABLED = "#6c5890"
BLOB_BODY = "#7c3aed"
BLOB_BODY_LIGHT = "#9b6dff"

# Fonts
FONT_FAMILY = "'Avenir Next', 'Avenir', 'Nunito', 'Quicksand', 'Helvetica Neue', sans-serif"
FONT_SIZE_TITLE = 24
FONT_SIZE_TAGLINE = 13
FONT_SIZE_CARD_TITLE = 14
FONT_SIZE_CARD_DESC = 11
FONT_SIZE_BUTTON = 14
FONT_SIZE_INFO = 12
FONT_SIZE_BLOB_LABEL = 12


def font_family() -> str:
    """Return the primary font family name for QFont usage."""
    return "Avenir Next"


def main_window_stylesheet() -> str:
    return f"""
        QMainWindow {{
            background-color: {BG_DEEP};
        }}
        QWidget {{
            background-color: transparent;
            color: {TEXT_PRIMARY};
            font-family: {FONT_FAMILY};
        }}
    """


def card_stylesheet(enabled: bool, selected: bool) -> str:
    if not enabled:
        return f"""
            background-color: {BG_CARD};
            border: 2px solid {TEXT_DISABLED};
            border-radius: 16px;
            color: {TEXT_DISABLED};
        """
    if selected:
        return f"""
            background-color: {BG_CARD};
            border: 2px solid {ACCENT_PRIMARY};
            border-radius: 16px;
            color: {TEXT_PRIMARY};
        """
    return f"""
        background-color: {BG_CARD};
        border: 2px solid transparent;
        border-radius: 16px;
        color: {TEXT_PRIMARY};
    """


def button_stylesheet(enabled: bool) -> str:
    if enabled:
        return f"""
            QPushButton {{
                background-color: {ACCENT_PRIMARY};
                color: {TEXT_PRIMARY};
                border: none;
                border-radius: 14px;
                font-family: {FONT_FAMILY};
                font-size: {FONT_SIZE_BUTTON}pt;
                font-weight: bold;
                letter-spacing: 2px;
                padding: 0 48px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_GLOW};
            }}
            QPushButton:pressed {{
                background-color: #6d28d9;
            }}
        """
    return f"""
        QPushButton {{
            background-color: {BG_CARD};
            color: {TEXT_DISABLED};
            border: none;
            border-radius: 14px;
            font-family: {FONT_FAMILY};
            font-size: {FONT_SIZE_BUTTON}pt;
            font-weight: bold;
            letter-spacing: 2px;
            padding: 0 48px;
        }}
    """
