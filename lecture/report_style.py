"""Assignment 結果報告的共用版面樣式。

版面以 Assignment #1 的結果報告為基準：粗體標題加全寬分隔線、粗體子標題、
縮排項目符號、淡藍表頭表格、圖下方說明文字，頁碼置於右下角。

由各作業資料夾的 build_report.py 匯入使用。
"""

from pathlib import Path
import unicodedata

import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend 必須先設定)

# 數學式以 Computer Modern 呈現（經典 LaTeX 外觀），與 Assignment #1 的報告一致。
# matplotlib 的預設值是 dejavusans，屬無襯線體，兩者外觀差異明顯。
matplotlib.rcParams["mathtext.fontset"] = "cm"


BOLD_FONT_PATH = "/System/Library/Fonts/STHeiti Medium.ttc"
BODY_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

BOLD = font_manager.FontProperties(fname=BOLD_FONT_PATH)
BODY = font_manager.FontProperties(fname=BODY_FONT_PATH)

PAGE_SIZE = (8.27, 11.69)  # A4 英吋
LEFT = 0.085
RIGHT = 0.915
USABLE_POINTS = (RIGHT - LEFT) * PAGE_SIZE[0] * 72

TITLE_SIZE = 22
SUBTITLE_SIZE = 19
HEADING_SIZE = 17
SUBHEADING_SIZE = 13.5
BODY_SIZE = 10.5
CAPTION_SIZE = 10
MONO_SIZE = 9

INK = "#1a1a1a"
MUTED = "#555555"
RULE = "#333333"
TABLE_HEADER_BG = "#eaf1fb"
TABLE_TEXT = "#1f4e99"
TABLE_LINE = "#b9cbe6"

LINE_STEP = 0.0205  # 一行文字佔的頁面高度比例
BULLET_STEP = 0.0265

# 中文排版禁則：這些標點不得出現在行首。
NO_LINE_START = "、。，；：！？）」』〉》〕｝］%,.;:!?)]}"


def text_units(text):
    """以「全形字 = 1 欄」計算文字寬度；半形字算 0.5 欄。"""
    total = 0.0
    for char in text:
        if unicodedata.east_asian_width(char) in ("F", "W"):
            total += 1.0
        else:
            total += 0.5
    return total


def wrap_text(text, max_units):
    """依欄寬換行，中文可在任意字之間斷行，英文單字保持完整。"""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue

        # 先切成「可斷行的單位」：CJK 逐字，連續的西文視為一個單字。
        tokens = []
        buffer = ""
        for char in paragraph:
            if unicodedata.east_asian_width(char) in ("F", "W"):
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(char)
            elif char == " ":
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(" ")
            else:
                buffer += char
        if buffer:
            tokens.append(buffer)

        current = ""
        for token in tokens:
            candidate = current + token
            if current and text_units(candidate) > max_units:
                # 禁則：若換行後該 token 會落在行首且為禁止起始的標點，
                # 就讓它留在本行（容許些微超寬），而非推到下一行開頭。
                if token[:1] in NO_LINE_START:
                    lines.append(candidate.rstrip())
                    current = ""
                    continue
                lines.append(current.rstrip())
                current = "" if token == " " else token
            else:
                current = candidate
        if current.strip() or not lines:
            lines.append(current.rstrip())
    return lines


class Report:
    """以 matplotlib PdfPages 逐頁繪製 A4 報告。"""

    def __init__(self, metadata=None):
        self.pages = []
        self.figure = None
        self.y = 0.0
        self.page_number = 0
        self.metadata = metadata or {}

    # --- 頁面管理 -------------------------------------------------
    def _new_figure(self):
        figure = plt.figure(figsize=PAGE_SIZE)
        figure.patch.set_facecolor("white")
        self.figure = figure
        self.pages.append(figure)
        return figure

    def _write(self, x, y, text, size, prop, color=INK, ha="left", va="top"):
        self.figure.text(x, y, text, fontsize=size, fontproperties=prop, color=color, ha=ha, va=va)

    def _page_number(self):
        self.figure.text(
            RIGHT, 0.043, str(self.page_number), fontsize=9, fontproperties=BODY,
            color=MUTED, ha="right", va="center",
        )

    def cover(self, title, subtitle, series, identity, info_lines):
        self._new_figure()
        self._write(0.5, 0.64, title, TITLE_SIZE, BOLD, ha="center", va="center")
        self._write(0.5, 0.565, subtitle, SUBTITLE_SIZE, BOLD, ha="center", va="center")
        self._write(0.5, 0.478, series, 12, BODY, color="#333333", ha="center", va="center")
        self.figure.add_artist(
            plt.Line2D([0.25, 0.75], [0.405, 0.405], color=RULE, linewidth=0.9)
        )
        self._write(0.5, 0.345, identity, 12.5, BOLD, ha="center", va="center")
        y = 0.283
        for line in info_lines:
            self._write(0.5, y, line, 11, BODY, color="#222222", ha="center", va="center")
            y -= 0.038
        return self

    def page(self, heading):
        self._new_figure()
        self.page_number += 1
        self._write(LEFT, 0.952, heading, HEADING_SIZE, BOLD)
        self.figure.add_artist(
            plt.Line2D([LEFT, RIGHT], [0.9235, 0.9235], color=RULE, linewidth=0.9)
        )
        self.y = 0.888
        self._page_number()
        return self

    # --- 內容元素 -------------------------------------------------
    def gap(self, amount=0.022):
        self.y -= amount
        return self

    def subheading(self, text):
        self._write(LEFT, self.y, text, SUBHEADING_SIZE, BOLD)
        self.y -= 0.036
        return self

    def text(self, body, size=BODY_SIZE, color=INK, indent=0.0):
        max_units = (USABLE_POINTS - indent * PAGE_SIZE[0] * 72) / size
        for line in wrap_text(body, max_units):
            self._write(LEFT + indent, self.y, line, size, BODY, color=color)
            self.y -= LINE_STEP
        return self

    def bullets(self, items, size=BODY_SIZE):
        marker_x = LEFT + 0.012
        text_x = LEFT + 0.048
        indent = text_x - LEFT
        max_units = (USABLE_POINTS - indent * PAGE_SIZE[0] * 72) / size
        for item in items:
            lines = wrap_text(item, max_units)
            self._write(marker_x, self.y, "•", size, BODY)
            for offset, line in enumerate(lines):
                self._write(text_x, self.y - offset * LINE_STEP, line, size, BODY)
            self.y -= BULLET_STEP + (len(lines) - 1) * LINE_STEP
        return self

    def math(self, expression, size=13, space=0.044):
        """置入 mathtext 數學式（維持 Computer Modern 字型，與 Assignment #1 一致）。

        分式與求和符號的實際高度遠大於字級，`space` 用來為較高的公式預留垂直空間，
        避免壓到下一段文字。
        """
        self.y -= 0.006
        self.figure.text(
            LEFT + 0.06, self.y, expression, fontsize=size, color=INK, ha="left", va="top"
        )
        self.y -= space
        return self

    def mono(self, block, size=MONO_SIZE):
        mono_prop = font_manager.FontProperties(family="monospace", size=size)
        for line in block.splitlines():
            self.figure.text(
                LEFT + 0.012, self.y, line, fontsize=size,
                fontproperties=mono_prop, color=INK, ha="left", va="top",
            )
            self.y -= 0.0155
        return self

    def table(self, headers, rows, widths, size=BODY_SIZE):
        """淡藍表頭、藍字內容、細框線的表格（沿用 Assignment #1 樣式）。"""
        total = sum(widths)
        span = RIGHT - LEFT
        edges = [LEFT]
        for width in widths:
            edges.append(edges[-1] + span * width / total)
        table_right = edges[-1]

        row_height = 0.0265
        header_top = self.y
        header_bottom = header_top - row_height

        self.figure.add_artist(
            plt.Rectangle(
                (LEFT, header_bottom), table_right - LEFT, row_height,
                facecolor=TABLE_HEADER_BG, edgecolor=TABLE_LINE, linewidth=0.7,
            )
        )
        for index, header in enumerate(headers):
            self._write(
                edges[index] + 0.012, header_top - row_height / 2, str(header),
                size, BODY, color=TABLE_TEXT, va="center",
            )

        y = header_bottom
        for row in rows:
            self.figure.add_artist(
                plt.Rectangle(
                    (LEFT, y - row_height), table_right - LEFT, row_height,
                    facecolor="white", edgecolor=TABLE_LINE, linewidth=0.7,
                )
            )
            for index, cell in enumerate(row):
                self._write(
                    edges[index] + 0.012, y - row_height / 2, str(cell),
                    size, BODY, color=TABLE_TEXT, va="center",
                )
            y -= row_height

        self.y = y - 0.028
        return self

    def figure_image(self, image_path, height=0.46, caption=None):
        image = plt.imread(str(image_path))
        top = self.y
        axes = self.figure.add_axes([LEFT, top - height, RIGHT - LEFT, height])
        axes.imshow(image)
        axes.axis("off")
        self.y = top - height - 0.032
        if caption:
            self.text(caption, size=CAPTION_SIZE, color="#222222")
        return self

    # --- 輸出 -----------------------------------------------------
    def save(self, output_path, dpi=150):
        output_path = Path(output_path)
        with PdfPages(output_path) as pdf:
            for figure in self.pages:
                pdf.savefig(figure, dpi=dpi)
            info = pdf.infodict()
            for key in ("Title", "Author", "Subject", "Creator"):
                if key in self.metadata:
                    info[key] = self.metadata[key]
        for figure in self.pages:
            plt.close(figure)
        self.pages = []
        return output_path
