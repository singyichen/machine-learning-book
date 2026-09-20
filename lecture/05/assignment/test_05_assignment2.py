"""Assignment #2 scaffold tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import pandas as pd


ASSIGNMENT_DIR = Path(__file__).resolve().parent
MODULE_PATH = ASSIGNMENT_DIR / "05_assignment2.py"


def load_assignment_module():
    spec = spec_from_file_location("assignment2", MODULE_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AssignmentDataTests(unittest.TestCase):
    def test_load_assignment_data_returns_expected_source_shapes(self):
        assignment2 = load_assignment_module()

        train_df, test_df = assignment2.load_assignment_data(ASSIGNMENT_DIR)

        self.assertEqual(train_df.shape, (15, 3))
        self.assertEqual(list(train_df.columns), ["x1", "x2", "y"])
        self.assertEqual(test_df.shape, (5, 3))
        self.assertEqual(list(test_df.columns), ["Wafer ID", "x1", "x2"])
        self.assertEqual(set(train_df["y"]), {0, 1})

    def test_validate_assignment_data_rejects_unexpected_columns(self):
        assignment2 = load_assignment_module()
        train_df = pd.DataFrame({"x1": [0.0], "x2": [1.0], "label": [1]})
        test_df = pd.DataFrame({"Wafer ID": ["Wafer A"], "x1": [0.0], "x2": [1.0]})

        with self.assertRaisesRegex(ValueError, "wat_train.csv columns"):
            assignment2.validate_assignment_data(train_df, test_df)


if __name__ == "__main__":
    unittest.main()
