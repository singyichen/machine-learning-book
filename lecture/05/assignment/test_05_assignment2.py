"""Assignment #2 tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
import unittest

import numpy as np
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


class LogisticRegressionTests(unittest.TestCase):
    def setUp(self):
        self.assignment2 = load_assignment_module()

    def test_sigmoid_is_stable_and_maps_zero_to_half(self):
        probabilities = self.assignment2.sigmoid(
            np.array([-1000.0, 0.0, 1000.0])
        )

        self.assertTrue(np.all(np.isfinite(probabilities)))
        np.testing.assert_allclose(probabilities, [0.0, 0.5, 1.0], atol=1e-12)

    def test_binary_cross_entropy_matches_hand_calculated_value(self):
        loss = self.assignment2.binary_cross_entropy(
            np.array([1.0, 0.0]), np.array([0.8, 0.25])
        )

        expected = -(np.log(0.8) + np.log(0.75)) / 2
        self.assertAlmostEqual(loss, expected, places=12)

    def test_gradient_descent_learns_a_linearly_separable_dataset(self):
        X = np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = self.assignment2.LogisticRegressionGD(
            learning_rate=0.1, epochs=300
        ).fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y.astype(int))
        self.assertLess(model.losses_[-1], model.losses_[0])


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.assignment2 = load_assignment_module()

    def test_feature_transformations_use_the_requested_x1_representation(self):
        frame = pd.DataFrame({"x1": [-2.0, 3.0], "x2": [0.5, 1.5]})

        raw = self.assignment2.build_features(frame, "raw")
        absolute = self.assignment2.build_features(frame, "abs")
        squared = self.assignment2.build_features(frame, "square")

        np.testing.assert_array_equal(raw, [[-2.0, 0.5], [3.0, 1.5]])
        np.testing.assert_array_equal(absolute, [[2.0, 0.5], [3.0, 1.5]])
        np.testing.assert_array_equal(squared, [[4.0, 0.5], [9.0, 1.5]])

    def test_standardizer_reuses_training_statistics_for_test_data(self):
        train = np.array([[0.0, 10.0], [2.0, 14.0]])
        test = np.array([[100.0, 200.0]])
        scaler = self.assignment2.Standardizer().fit(train)

        transformed = scaler.transform(test)

        np.testing.assert_array_equal(scaler.mean_, [1.0, 12.0])
        np.testing.assert_array_equal(scaler.scale_, [1.0, 2.0])
        np.testing.assert_array_equal(transformed, [[99.0, 94.0]])


class AssignmentPipelineTests(unittest.TestCase):
    def setUp(self):
        self.assignment2 = load_assignment_module()

    def test_training_comparison_includes_all_features_and_learning_rates(self):
        train_df, _ = self.assignment2.load_assignment_data(ASSIGNMENT_DIR)

        comparison, runs = self.assignment2.compare_training_options(
            train_df,
            transformations=("raw", "abs", "square"),
            learning_rates=(0.01, 0.1),
            epochs=300,
        )

        self.assertEqual(len(comparison), 6)
        self.assertEqual(set(comparison["Feature Transformation"]), {"raw", "abs", "square"})
        self.assertEqual(set(comparison["Learning Rate"]), {0.01, 0.1})
        self.assertEqual(len(runs), 6)

    def test_run_assignment_is_reproducible_and_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self.assignment2.run_assignment(
                data_dir=ASSIGNMENT_DIR,
                output_dir=first_dir,
                epochs=500,
            )
            second = self.assignment2.run_assignment(
                data_dir=ASSIGNMENT_DIR,
                output_dir=second_dir,
                epochs=500,
            )

            pd.testing.assert_frame_equal(first["predictions"], second["predictions"])
            self.assertEqual(
                first["predictions"].columns.tolist(),
                ["Wafer ID", "x1", "x2", "abs_x1", "Pass Probability", "Result"],
            )
            self.assertEqual(first["predictions"].shape, (5, 6))
            self.assertTrue(first["predictions"]["Pass Probability"].between(0, 1).all())
            self.assertTrue(set(first["predictions"]["Result"]).issubset({"Pass", "Fail"}))

            for filename in (
                "assignment2_loss_curves.png",
                "assignment2_decision_boundary.png",
                "assignment2_predictions.csv",
                "assignment2_results.txt",
            ):
                self.assertGreater((Path(first_dir) / filename).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
