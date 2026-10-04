"""Assignment 結果報告的共用版面樣式。

版面以**已繳交的** Assignment #1 結果報告（字體調整版，commit 77ecd01）為基準，
所有數值都是從該 PDF 實際量出來的：

- 字型：全文 Arial Unicode MS；封面主標題 Arial Bold；頁碼 Arial Regular
- 字級：封面 24 / 章節標題 20 / 子標題 16（次要 14）/ 內文 12 / 圖說 10 / 頁碼 9
- 顏色：內文純黑；表格文字與表頭框線 #003D9E
- 封面算第 1 頁但不印頁碼，「一、」從第 2 頁開始

注意 A1 的標題**不是**黑體中黑，而是放大的 Arial Unicode。早期版本誤用
STHeiti Medium，英文字母會變成細長的幾何字形，看起來完全不像 A1。

由各作業資料夾的 build_report.py 匯入使用。
"""

from pathlib import Path
import re
import unicodedata

import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend 必須先設定)

# 數學式以 Computer Modern 呈現（經典 LaTeX 外觀），與 Assignment #1 的報告一致。
# matplotlib 的預設值是 dejavusans，屬無襯線體，兩者外觀差異明顯。
matplotlib.rcParams["mathtext.fontset"] = "cm"

# 以 TrueType（Type 42）嵌入字型，與 A1 的 CID TrueType 相同。matplotlib 預設的
# Type 3 不帶中文的 ToUnicode 對照，PDF 裡的中文無法選取、搜尋或複製。
matplotlib.rcParams["pdf.fonttype"] = 42


FONT_DIR = "/System/Library/Fonts/Supplemental"
BODY_FONT_PATH = f"{FONT_DIR}/Arial Unicode.ttf"
TITLE_FONT_PATH = f"{FONT_DIR}/Arial Bold.ttf"  # 只有拉丁字母，封面主標題不可含中文
NUMBER_FONT_PATH = f"{FONT_DIR}/Arial.ttf"

BODY = font_manager.FontProperties(fname=BODY_FONT_PATH)
TITLE = font_manager.FontProperties(fname=TITLE_FONT_PATH)
NUMBER = font_manager.FontProperties(fname=NUMBER_FONT_PATH)

PAGE_SIZE = (8.27, 11.69)  # A4 英吋
PAGE_WIDTH = PAGE_SIZE[0] * 72
PAGE_HEIGHT = PAGE_SIZE[1] * 72

# 以下位置皆為「距頁面左緣 / 上緣的點數」，與 A1 量測值直接對應。
LEFT_PT = 48
RIGHT_PT = 548
BULLET_MARK_PT = 60
BULLET_TEXT_PT = 80
MATH_PT = 83
TABLE_LEFT_PT = 72
TABLE_RIGHT_PT = 524
PAGE_NUMBER_RIGHT_PT = 535
PAGE_NUMBER_BASELINE_PT = 817

TITLE_SIZE = 24
SUBTITLE_SIZE = 24
HEADING_SIZE = 20
SUBHEADING_SIZE = 16
MINOR_SUBHEADING_SIZE = 14
BODY_SIZE = 12
CAPTION_SIZE = 10
PAGE_NUMBER_SIZE = 9
MONO_SIZE = 9

INK = "#000000"
PAGE_NUMBER_INK = "#616161"
HEADING_RULE = "#2e2e2e"
COVER_RULE = "#474747"
TABLE_TEXT = "#003d9e"
TABLE_HEADER_BG = "#f0f7ff"
TABLE_OUTLINE = "#b8c2d1"
TABLE_COLUMN_LINE = "#c7d1de"
TABLE_ROW_LINE = "#dbdbdb"

LINE_STEP_PT = 18
CAPTION_STEP_PT = 14
BULLET_STEP_PT = 26
TABLE_ROW_PT = 22

# 中文排版禁則：這些標點不得出現在行首。
NO_LINE_START = "、。，；：！？）」』〉》〕｝］%,.;:!?)]}"


def fraction(points_from_top):
    """距上緣的點數 → matplotlib figure 座標（下緣為 0）。"""
    return 1 - points_from_top / PAGE_HEIGHT


def x_fraction(points_from_left):
    return points_from_left / PAGE_WIDTH


LEFT = x_fraction(LEFT_PT)
RIGHT = x_fraction(RIGHT_PT)
USABLE_POINTS = RIGHT_PT - LEFT_PT

# 頁碼基線在 817 pt；游標（下一行的基線）超過 800 pt 就會壓到頁碼。
BOTTOM_LIMIT = fraction(800)


class PageOverflow(RuntimeError):
    """內容超出頁面可用高度。把該頁拆開，或縮減內容。"""


MATH_SPAN = re.compile(r"\$[^$]*\$")
MATH_SCRIPT = re.compile(r"[_^](\{[^}]*\}|\\[A-Za-z]+|.)")
MATH_MARKUP = re.compile(r"\\[A-Za-z]+|[{}$\s]")


def _math_units(span):
    """估算 mathtext 片段渲染後的寬度。

    `\\dfrac`、`\\mathrm` 這類指令與 `{}` 都不佔版面；上下標縮小排版，
    每個可見符號約 0.3 欄，其餘可見符號與半形西文相同算 0.5 欄。
    """
    scripts = MATH_SCRIPT.findall(span)
    script_glyphs = sum(max(len(MATH_MARKUP.sub("", part)), 1) for part in scripts)
    body = MATH_MARKUP.sub("", MATH_SCRIPT.sub("", span))
    return max(0.5 * len(body) + 0.3 * script_glyphs, 0.5)


def text_units(text):
    """以「全形字 = 1 欄」計算文字寬度；半形字算 0.5 欄，`$...$` 依可見符號估算。"""
    total = 0.0
    position = 0
    for match in MATH_SPAN.finditer(text):
        total += _plain_units(text[position:match.start()])
        total += _math_units(match.group())
        position = match.end()
    total += _plain_units(text[position:])
    return total


def _plain_units(text):
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
        chars = iter(enumerate(paragraph))
        for index, char in chars:
            if char == "$" and "$" in paragraph[index + 1:]:
                # 整段 $...$ 是一個不可拆的 token：拆開會讓 matplotlib 看到單數個 $。
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                end = paragraph.index("$", index + 1)
                tokens.append(paragraph[index:end + 1])
                for _ in range(end - index):
                    next(chars)
                continue
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
    """以 matplotlib PdfPages 逐頁繪製 A4 報告。

    `self.y` 是下一行文字的基線位置（figure 座標），各元素由上往下推進。
    """

    def __init__(self, metadata=None):
        self.pages = []
        self.figure = None
        self.y = 0.0
        self.page_number = 0
        self.page_heading = None
        self.metadata = metadata or {}

    # --- 頁面管理 -------------------------------------------------
    def _new_figure(self):
        figure = plt.figure(figsize=PAGE_SIZE)
        figure.patch.set_facecolor("white")
        self.figure = figure
        self.pages.append(figure)
        self.page_number += 1
        return figure

    def _write(self, x, y, text, size, prop=BODY, color=INK, ha="left", va="baseline"):
        self.figure.text(x, y, text, fontsize=size, fontproperties=prop, color=color, ha=ha, va=va)

    def _line(self, x0, x1, y0, y1, color, width):
        self.figure.add_artist(plt.Line2D([x0, x1], [y0, y1], color=color, linewidth=width))

    def _advance(self, points):
        self.y -= points / PAGE_HEIGHT

    def _check_room(self):
        """游標低於底線代表內容已溢出——靜默裁切很難發現，直接報錯。"""
        if self.page_heading is not None and self.y < BOTTOM_LIMIT:
            raise PageOverflow(
                f"「{self.page_heading}」這一頁的內容超出可用高度"
                f"（游標 {self.y:.3f} < 底線 {BOTTOM_LIMIT:.3f}）。請把這頁拆成兩頁。"
            )

    def cover(self, title, subtitle, series, identity, info_lines):
        """封面算第 1 頁，但和 A1 一樣不印頁碼。"""
        self._new_figure()
        self.page_heading = None
        center = 0.5
        self._write(center, fraction(318), title, TITLE_SIZE, TITLE, ha="center")
        self._write(center, fraction(378), subtitle, SUBTITLE_SIZE, ha="center")
        self._write(center, fraction(446), series, 14, color="#404040", ha="center")
        self._line(x_fraction(149), x_fraction(446), fraction(505), fraction(505), COVER_RULE, 0.6)
        self._write(center, fraction(558), identity, 14, ha="center")
        for index, line in enumerate(info_lines):
            self._write(center, fraction(594 + 32 * index), line, 12, color="#333333", ha="center")
        return self

    def page(self, heading):
        self._new_figure()
        self.page_heading = heading
        self._write(LEFT, fraction(59), heading, HEADING_SIZE)
        self._line(LEFT, RIGHT, fraction(69), fraction(69), HEADING_RULE, 0.8)
        self._write(
            x_fraction(PAGE_NUMBER_RIGHT_PT), fraction(PAGE_NUMBER_BASELINE_PT),
            str(self.page_number), PAGE_NUMBER_SIZE, NUMBER, color=PAGE_NUMBER_INK, ha="right",
        )
        self.y = fraction(113)
        return self

    # --- 內容元素 -------------------------------------------------
    def gap(self, amount=0.022):
        """額外留白，單位為頁面高度比例（0.03 ≈ 25 pt）。"""
        self.y -= amount
        return self

    def subheading(self, text, level=1):
        """level=1 為 16 pt（作業說明、總結等頁的段落標題）；
        level=2 為 14 pt（數學觀念的編號項目、結果頁內的小節）。"""
        if level == 1:
            self._write(LEFT, self.y, text, SUBHEADING_SIZE)
            self._advance(32)
        else:
            self._write(LEFT, self.y, text, MINOR_SUBHEADING_SIZE)
            self._advance(30)
        self._check_room()
        return self

    def text(self, body, size=BODY_SIZE, color=INK, indent=0.0):
        step = LINE_STEP_PT if size >= BODY_SIZE else CAPTION_STEP_PT
        max_units = (USABLE_POINTS - indent * PAGE_WIDTH) / size
        for line in wrap_text(body, max_units):
            self._write(LEFT + indent, self.y, line, size, color=color)
            self._advance(step)
        self._check_room()
        return self

    def bullets(self, items, size=BODY_SIZE):
        max_units = (RIGHT_PT - BULLET_TEXT_PT) / size
        for item in items:
            lines = wrap_text(item, max_units)
            self._write(x_fraction(BULLET_MARK_PT), self.y, "•", size)
            for offset, line in enumerate(lines):
                y = self.y - offset * LINE_STEP_PT / PAGE_HEIGHT
                self._write(x_fraction(BULLET_TEXT_PT), y, line, size)
            self._advance(BULLET_STEP_PT + (len(lines) - 1) * LINE_STEP_PT)
        self._check_room()
        return self

    def math(self, expression, size=14, space=0.044):
        """置入 mathtext 數學式（維持 Computer Modern 字型，與 Assignment #1 一致）。

        分式與求和符號的實際高度遠大於字級，`space` 用來為較高的公式預留垂直空間，
        避免壓到下一段文字。
        """
        top = self.y + 10 / PAGE_HEIGHT
        self.figure.text(
            x_fraction(MATH_PT), top, expression, fontsize=size, color=INK, ha="left", va="top"
        )
        self.y = top - space
        self._check_room()
        return self

    def mono(self, block, size=MONO_SIZE):
        mono_prop = font_manager.FontProperties(family="monospace", size=size)
        for line in block.splitlines():
            self._write(x_fraction(BULLET_MARK_PT), self.y, line, size, mono_prop)
            self._advance(13)
        self._check_room()
        return self

    def table(self, headers, rows, widths, size=BODY_SIZE):
        """淡藍表頭、藍字內容、細框線的表格（量自 Assignment #1 第 5 頁）。"""
        total = sum(widths)
        span = TABLE_RIGHT_PT - TABLE_LEFT_PT
        edges = [TABLE_LEFT_PT]
        for width in widths:
            edges.append(edges[-1] + span * width / total)

        top = self.y * PAGE_HEIGHT  # 以「距下緣點數」計算，方便畫矩形
        top = PAGE_HEIGHT - top - 4  # 換成距上緣點數；表格上緣略高於游標基線
        left, right = x_fraction(TABLE_LEFT_PT), x_fraction(edges[-1])
        row_height = TABLE_ROW_PT / PAGE_HEIGHT
        bottom = top + TABLE_ROW_PT * (len(rows) + 1)

        self.figure.add_artist(plt.Rectangle(
            (left, fraction(top + TABLE_ROW_PT)), right - left, row_height,
            facecolor=TABLE_HEADER_BG, edgecolor=TABLE_TEXT, linewidth=0.8,
        ))
        for edge in edges[1:-1]:
            self._line(x_fraction(edge), x_fraction(edge), fraction(top), fraction(bottom),
                       TABLE_COLUMN_LINE, 0.5)
        for index in range(2, len(rows) + 2):
            y = fraction(top + TABLE_ROW_PT * index)
            self._line(left, right, y, y, TABLE_ROW_LINE, 0.45)
        self.figure.add_artist(plt.Rectangle(
            (left, fraction(bottom)), right - left, fraction(top) - fraction(bottom),
            facecolor="none", edgecolor=TABLE_OUTLINE, linewidth=0.6,
        ))

        for row_index, cells in enumerate([headers, *rows]):
            baseline = fraction(top + TABLE_ROW_PT * row_index + 16)
            for column, cell in enumerate(cells):
                self._write(x_fraction(edges[column] + 7), baseline, str(cell), size, color=TABLE_TEXT)

        self.y = fraction(bottom + 40)
        self._check_room()
        return self

    def figure_image(self, image_path, height=0.46, caption=None):
        image = plt.imread(str(image_path))
        top = self.y + 10 / PAGE_HEIGHT
        axes = self.figure.add_axes([LEFT, top - height, RIGHT - LEFT, height])
        axes.imshow(image)
        axes.axis("off")
        self.y = top - height - 20 / PAGE_HEIGHT
        if caption:
            self.text(caption, size=CAPTION_SIZE)
        self._check_room()
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
