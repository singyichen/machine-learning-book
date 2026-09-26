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


if __name__ == "__main__":
    unittest.main()
