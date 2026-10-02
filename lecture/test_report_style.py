"""Tests for the shared Assignment report style."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_style


class WrapTextTests(unittest.TestCase):
    """Overflowing the right margin is the break these tests catch."""

    def test_ascii_text_wraps_at_the_column_budget(self):
        lines = report_style.wrap_text("word " * 12, max_units=20)

        for line in lines:
            self.assertLessEqual(report_style.text_units(line), 20)
        self.assertEqual(" ".join(lines), ("word " * 12).strip())

    def test_cjk_counts_as_a_full_column_and_ascii_as_half(self):
        self.assertEqual(report_style.text_units("機器學習"), 4.0)
        self.assertEqual(report_style.text_units("abcd"), 2.0)
        self.assertEqual(report_style.text_units("機器 ab"), 3.5)

    def test_cjk_run_without_spaces_still_wraps(self):
        # A 60-character Chinese sentence has no spaces to break on; a naive
        # whitespace wrapper would emit one 60-unit line and run off the page.
        sentence = "測" * 60

        lines = report_style.wrap_text(sentence, max_units=20)

        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertLessEqual(report_style.text_units(line), 20)
        self.assertEqual("".join(lines), sentence)

    def test_mixed_cjk_and_latin_keeps_latin_words_intact(self):
        text = "使用 LogisticRegression 建立分類器並輸出結果"

        lines = report_style.wrap_text(text, max_units=12)

        for line in lines:
            self.assertLessEqual(report_style.text_units(line), 12)
        self.assertIn("LogisticRegression", "".join(lines))

    def test_no_line_starts_with_closing_punctuation(self):
        # 中文排版禁則：、。，）等標點不得出現在行首。
        # 以會讓「、」剛好落在行首的字串驗證。
        text = "完成資料切分、特徵縮放、L1 特徵選取、交叉驗證模型選取與模型評估"

        for budget in range(8, 30):
            lines = report_style.wrap_text(text, max_units=budget)
            for line in lines:
                self.assertNotIn(
                    line[:1],
                    report_style.NO_LINE_START,
                    f"budget={budget} produced a line starting with {line[:1]!r}",
                )

    def test_explicit_newlines_start_new_lines(self):
        lines = report_style.wrap_text("第一行\n第二行", max_units=40)

        self.assertEqual(lines, ["第一行", "第二行"])


class OverflowGuardTests(unittest.TestCase):
    """內容超出頁面底部時會被靜默裁掉，必須主動報錯而不是默默產出壞頁面。"""

    def test_page_overflow_raises_instead_of_silently_clipping(self):
        report = report_style.Report()
        report.page("測試")

        with self.assertRaisesRegex(report_style.PageOverflow, "測試"):
            for _ in range(80):
                report.text("填滿頁面用的文字。")

    def test_a_page_within_its_budget_does_not_raise(self):
        report = report_style.Report()
        report.page("測試")

        for _ in range(20):
            report.text("填滿頁面用的文字。")

        self.assertGreater(report.y, report_style.BOTTOM_LIMIT)


class MathRenderingTests(unittest.TestCase):
    """公式必須以 Computer Modern 呈現，才會與 Assignment #1 的報告一致。"""

    def test_formulas_embed_computer_modern_not_the_matplotlib_default(self):
        import tempfile

        import fitz

        report = report_style.Report()
        report.page("測試")
        report.math(r"$z = \dfrac{x - \mu}{\sigma}$", size=14)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "math.pdf"
            report.save(output)

            document = fitz.open(output)
            fonts = {font[3].split("+")[-1] for font in document[0].get_fonts()}
            document.close()

        # matplotlib 預設的 mathtext.fontset 是 dejavusans，與 Assignment #1
        # 的 Cmr10 / Cmmi10 / Cmsy10 外觀完全不同。
        self.assertTrue(
            any(name.lower().startswith("cm") for name in fonts),
            f"公式未使用 Computer Modern，實際字型：{sorted(fonts)}",
        )


def render_spans(report):
    """把報告存成 PDF，回傳 [(頁碼, 字型名稱, 字級, 顏色, 文字), ...]。"""
    import tempfile

    import fitz

    spans = []
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "spans.pdf"
        report.save(output)
        document = fitz.open(output)
        for page_index, page in enumerate(document):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        if span["text"].strip():
                            spans.append((
                                page_index + 1,
                                span["font"].split("+")[-1],
                                round(span["size"], 1),
                                span["color"],
                                span["text"],
                            ))
        document.close()
    return spans


def find_span(spans, text):
    matches = [span for span in spans if text in span[4]]
    if not matches:
        raise AssertionError(f"PDF 中找不到 {text!r}；實際文字：{[s[4] for s in spans]}")
    return matches[0]


class Assignment1TypographyTests(unittest.TestCase):
    """數值取自已繳交的 Assignment #1 報告（字體調整版，commit 77ecd01）。

    版面結構對了不代表字體對了：A2 曾經以黑體中黑繪製標題、內文 10.5 pt，
    肉眼一看就和 A1 不同，但「有 Cmr10 與 STHeiti」的字型檢查照樣通過。
    """

    @classmethod
    def setUpClass(cls):
        report = report_style.Report()
        report.cover(
            "Assignment #9", "測試副標題", "機器學習實作系列　第 9 週：測試",
            "姓名：測試", ["資料集：test.csv"],
        )
        report.page("一、作業說明")
        report.subheading("目的")
        report.subheading("1. 次要標題", level=2)
        report.text("內文 Body text")
        report.table(["欄位"], [("表格內容",)], [1.0])
        cls.spans = render_spans(report)

    def test_cover_title_uses_real_arial_bold(self):
        _, font, size, _, _ = find_span(self.spans, "Assignment")
        self.assertEqual(font, "Arial-BoldMT")
        self.assertEqual(size, 24.0)

    def test_cover_subtitle_matches_title_size_in_regular_weight(self):
        _, font, size, _, _ = find_span(self.spans, "測試副標題")
        self.assertEqual(font, "ArialUnicodeMS")
        self.assertEqual(size, 24.0)

    def test_heading_sizes_follow_assignment1(self):
        expected = {"一、作業說明": 20.0, "目的": 16.0, "1. 次要標題": 14.0, "內文 Body text": 12.0}
        for text, size in expected.items():
            with self.subTest(text=text):
                _, font, actual, color, _ = find_span(self.spans, text)
                self.assertEqual(font, "ArialUnicodeMS")
                self.assertEqual(actual, size)
                self.assertEqual(color, 0x000000)

    def test_no_heiti_font_anywhere(self):
        # A1 的標題是放大的 Arial Unicode，不是黑體中黑。
        fonts = {span[1] for span in self.spans}
        self.assertFalse(any("Heiti" in font for font in fonts), sorted(fonts))

    def test_table_text_is_assignment1_blue(self):
        _, _, size, color, _ = find_span(self.spans, "表格內容")
        self.assertEqual(size, 12.0)
        self.assertEqual(color, 0x003D9E)

    def test_cover_counts_as_page_one(self):
        # A1 的「一、作業說明」頁碼為 2；封面本身不印頁碼。
        page_numbers = [s for s in self.spans if s[4].strip().isdigit() and s[2] == 9.0]
        self.assertEqual([(s[0], s[4]) for s in page_numbers], [(2, "2")])
        self.assertEqual(page_numbers[0][1], "ArialMT")


if __name__ == "__main__":
    unittest.main()
