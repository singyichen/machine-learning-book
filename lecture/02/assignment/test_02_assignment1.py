"""Assignment #1 tests for the originally submitted script."""

from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ASSIGNMENT_DIR = Path(__file__).resolve().parent
MODULE_PATH = ASSIGNMENT_DIR / "02_assignment1.py"
DATA_PATH = ASSIGNMENT_DIR / "tech_employees.csv"


def load_assignment_module():
    spec = spec_from_file_location("assignment1", MODULE_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_assignment_script(output_dir):
    output_path = Path(output_dir)
    shutil.copyfile(DATA_PATH, output_path / DATA_PATH.name)
    matplotlib_config = output_path / ".matplotlib"
    matplotlib_config.mkdir()
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(matplotlib_config)
    environment["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    return subprocess.run(
        [sys.executable, str(MODULE_PATH)],
        cwd=output_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


class AssignmentDataTests(unittest.TestCase):
    def test_source_data_has_expected_shape_columns_and_labels(self):
        frame = pd.read_csv(DATA_PATH)

        self.assertEqual(frame.shape, (300, 5))
        self.assertEqual(
            list(frame.columns),
            [
                "Language_Skill",
                "Math_Skill",
                "Coding_Skill",
                "Health_Condition",
                "Performance_Label",
            ],
        )
        self.assertFalse(frame.isna().any().any())
        self.assertTrue(
            all(
                pd.api.types.is_numeric_dtype(frame[column])
                for column in frame.columns
            )
        )
        self.assertTrue(
            np.isfinite(frame.to_numpy(dtype=float)).all()
        )
        self.assertEqual(set(frame["Performance_Label"]), {0, 1})
        self.assertEqual(set(frame.iloc[:200]["Performance_Label"]), {0, 1})
        self.assertEqual(set(frame.iloc[200:300]["Performance_Label"]), {0, 1})
        feature_columns = frame.columns[:-1]
        self.assertTrue((frame.iloc[:200][feature_columns].std() > 0).all())


class AdalineSGDTests(unittest.TestCase):
    def setUp(self):
        self.assignment1 = load_assignment_module()

    def test_update_weights_matches_squared_error_gradient(self):
        model = self.assignment1.AdalineSGD()
        model.w_ = np.array([0.2, -0.1])
        model.b_ = 0.05

        loss = model._update_weights(
            np.array([2.0, 3.0]), target=1.0, eta=0.1
        )

        np.testing.assert_allclose(model.w_, [0.54, 0.41])
        self.assertAlmostEqual(model.b_, 0.22)
        self.assertAlmostEqual(loss, 0.7225)

    def test_fit_learns_linearly_separable_examples_and_reduces_loss(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])

        model = self.assignment1.AdalineSGD(
            eta=0.01,
            n_iter=100,
            shuffle=True,
            adaptive=True,
            random_state=1,
        ).fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)
        self.assertLess(model.losses_[-1], model.losses_[0])

    def test_predict_uses_half_as_the_classification_threshold(self):
        model = self.assignment1.AdalineSGD()
        model.w_ = np.array([1.0])
        model.b_ = 0.0

        predictions = model.predict(np.array([[0.49], [0.5]]))

        np.testing.assert_array_equal(predictions, [0, 1])

    def test_fit_is_reproducible_for_the_same_random_state(self):
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        parameters = {
            "eta": 0.01,
            "n_iter": 20,
            "shuffle": True,
            "adaptive": True,
            "random_state": 7,
        }

        first = self.assignment1.AdalineSGD(**parameters).fit(X, y)
        second = self.assignment1.AdalineSGD(**parameters).fit(X, y)

        np.testing.assert_array_equal(first.w_, second.w_)
        self.assertEqual(first.b_, second.b_)
        np.testing.assert_array_equal(first.losses_, second.losses_)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.assignment1 = load_assignment_module()

    def test_standard_scaler_reuses_training_statistics_for_test_data(self):
        train = np.array([[0.0, 10.0], [2.0, 14.0]])
        test = np.array([[100.0, 200.0]])
        scaler = self.assignment1.StandardScaler().fit(train)

        transformed = scaler.transform(test)

        np.testing.assert_array_equal(scaler.mean_, [1.0, 12.0])
        np.testing.assert_array_equal(scaler.std_, [1.0, 2.0])
        np.testing.assert_array_equal(transformed, [[99.0, 94.0]])

    def test_train_and_evaluate_fits_scaler_on_training_data_only(self):
        X_train = np.array([[0.0, 10.0], [2.0, 14.0]])
        y_train = np.array([0, 1])
        X_test = np.array([[100.0, 200.0], [102.0, 204.0]])
        y_test = np.array([1, 1])
        original_scaler = self.assignment1.StandardScaler

        class TrackingStandardScaler(original_scaler):
            instances = []

            def __init__(self):
                self.fit_inputs = []
                self.transform_inputs = []
                self.instances.append(self)

            def fit(self, X):
                self.fit_inputs.append(np.array(X, copy=True))
                return super().fit(X)

            def transform(self, X):
                self.transform_inputs.append(np.array(X, copy=True))
                return super().transform(X)

        with patch.object(
            self.assignment1, "StandardScaler", TrackingStandardScaler
        ):
            _, scaler, _ = self.assignment1.train_and_evaluate(
                X_train,
                y_train,
                X_test,
                y_test,
                n_iter=1,
                random_state=1,
            )

        self.assertEqual(len(TrackingStandardScaler.instances), 1)
        self.assertIs(scaler, TrackingStandardScaler.instances[0])
        self.assertEqual(len(scaler.fit_inputs), 1)
        self.assertEqual(len(scaler.transform_inputs), 2)
        np.testing.assert_array_equal(scaler.fit_inputs[0], X_train)
        np.testing.assert_array_equal(scaler.transform_inputs[0], X_train)
        np.testing.assert_array_equal(scaler.transform_inputs[1], X_test)
        np.testing.assert_array_equal(scaler.mean_, [1.0, 12.0])
        np.testing.assert_array_equal(scaler.std_, [1.0, 2.0])


class AssignmentScriptTests(unittest.TestCase):
    def test_original_script_is_reproducible_and_writes_expected_artifacts(self):
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            first = run_assignment_script(first_dir)
            second = run_assignment_script(second_dir)

            self.assertEqual(
                first.returncode,
                0,
                msg=f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}",
            )
            self.assertEqual(
                second.returncode,
                0,
                msg=f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}",
            )
            self.assertEqual(first.stdout, second.stdout)

            expected_report = "\n".join((
                "Feature Combination                     Test Accuracy",
                "-----------------------------------------------------",
                "Math_Skill & Coding_Skill                        0.84",
                "All Four Features                                0.83",
                "Language_Skill & Coding_Skill                    0.79",
                "Coding_Skill & Health_Condition                  0.75",
                "Language_Skill & Math_Skill                      0.61",
                "Language_Skill & Health_Condition                0.61",
                "Math_Skill & Health_Condition                    0.61",
                "",
                "結論：以「包含該特徵之組合平均正確率」排序，"
                "Coding_Skill (0.80)、Math_Skill (0.72)、"
                "Language_Skill (0.71)、Health_Condition (0.70)。"
                "Coding_Skill 鑑別度最高，Math_Skill 次之。",
            ))
            self.assertEqual(first.stdout, expected_report + "\n")

            for filename in (
                "assignment1_results.txt",
                "assignment1_decision_boundaries.png",
                "assignment1_loss_curves.png",
                "assignment1_adaptive_vs_fixed.png",
            ):
                with self.subTest(filename=filename):
                    first_artifact = Path(first_dir) / filename
                    second_artifact = Path(second_dir) / filename
                    first_diagnostic = (
                        f"missing {filename} from first run\n"
                        f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}"
                    )
                    second_diagnostic = (
                        f"missing {filename} from second run\n"
                        f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}"
                    )
                    self.assertTrue(
                        first_artifact.is_file(), msg=first_diagnostic
                    )
                    self.assertTrue(
                        second_artifact.is_file(), msg=second_diagnostic
                    )
                    self.assertGreater(first_artifact.stat().st_size, 0)
                    self.assertGreater(second_artifact.stat().st_size, 0)

            first_report = (
                Path(first_dir) / "assignment1_results.txt"
            ).read_text(encoding="utf-8")
            second_report = (
                Path(second_dir) / "assignment1_results.txt"
            ).read_text(encoding="utf-8")
            self.assertEqual(first_report, second_report)
            self.assertEqual(first_report, expected_report + "\n")


if __name__ == "__main__":
    unittest.main()
