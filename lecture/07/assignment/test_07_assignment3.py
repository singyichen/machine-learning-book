"""Assignment #3 tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import ast
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ASSIGNMENT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = ASSIGNMENT_DIR / "07_assignment3_train.py"
EVAL_PATH = ASSIGNMENT_DIR / "07_assignment3_eval.py"


def load_module(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_train_module():
    return load_module(TRAIN_PATH, "assignment3_train")


def load_eval_module():
    return load_module(EVAL_PATH, "assignment3_eval")


_TRAINED_MODEL_DIR = None


def trained_model_path():
    """Train the real pipeline once and reuse the artefact across test classes."""
    global _TRAINED_MODEL_DIR
    if _TRAINED_MODEL_DIR is None:
        _TRAINED_MODEL_DIR = tempfile.mkdtemp(prefix="assignment3_model_")
        train = load_train_module()
        train.run_training(output_dir=_TRAINED_MODEL_DIR)
    return Path(_TRAINED_MODEL_DIR) / "digits_pipeline_lr.pkl"


class SplitContractTests(unittest.TestCase):
    """The grader re-splits the data, so both scripts must agree exactly."""

    def test_split_holds_out_360_samples_of_the_1797_digit_images(self):
        train = load_train_module()

        X_train, X_test, y_train, y_test = train.split_train_test(*train.load_digit_images())

        self.assertEqual(X_train.shape, (1437, 64))
        self.assertEqual(X_test.shape, (360, 64))
        self.assertEqual(y_train.shape, (1437,))
        self.assertEqual(y_test.shape, (360,))

    def test_split_is_pinned_to_seed_42_so_the_grader_sees_the_same_test_set(self):
        train = load_train_module()
        X, y = train.load_digit_images()

        _, _, _, y_test = train.split_train_test(X, y)

        # Hand-derived from train_test_split(..., test_size=0.2, random_state=42).
        # A changed seed or split ratio reshuffles these labels.
        np.testing.assert_array_equal(y_test[:10], [6, 9, 3, 7, 2, 1, 5, 2, 5, 2])


class FeatureBudgetTests(unittest.TestCase):
    """The assignment caps the model at 44 of the 64 pixels."""

    def test_validate_feature_budget_rejects_a_mask_over_the_44_pixel_cap(self):
        train = load_train_module()
        mask = np.zeros(64, dtype=bool)
        mask[:45] = True

        with self.assertRaisesRegex(ValueError, "45"):
            train.validate_feature_budget(mask)

    def test_validate_feature_budget_accepts_a_mask_at_the_44_pixel_cap(self):
        train = load_train_module()
        mask = np.zeros(64, dtype=bool)
        mask[:44] = True

        self.assertIs(train.validate_feature_budget(mask), mask)


class TrainedPipelineTests(unittest.TestCase):
    """The pipeline is trained once and shared, because fitting is slow."""

    @classmethod
    def setUpClass(cls):
        cls.train = load_train_module()
        X, y = cls.train.load_digit_images()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = cls.train.split_train_test(X, y)
        cls.search = cls.train.select_best_model(cls.X_train, cls.y_train)
        cls.pipeline = cls.search.best_estimator_

    def test_feature_selection_drops_at_least_20_of_the_64_pixels(self):
        mask = self.train.extract_feature_mask(self.pipeline)

        self.assertEqual(mask.shape, (64,))
        self.assertLessEqual(int(mask.sum()), 44)

    def test_classifier_is_fitted_on_only_the_selected_pixels(self):
        mask = self.train.extract_feature_mask(self.pipeline)

        classifier = self.pipeline.named_steps["classifier"]

        # A mask read off the wrong step would not match the classifier's input width.
        self.assertEqual(classifier.coef_.shape[1], int(mask.sum()))

    def test_test_set_accuracy_clears_the_95_percent_requirement(self):
        accuracy = (self.pipeline.predict(self.X_test) == self.y_test).mean()

        self.assertGreaterEqual(accuracy, 0.95)

    def test_saved_model_reloads_and_predicts_identically(self):
        import joblib
        import tempfile

        with tempfile.TemporaryDirectory() as output_dir:
            model_path = self.train.save_pipeline(self.pipeline, output_dir)
            reloaded = joblib.load(model_path)

            # Dumping the bare classifier instead of the pipeline would drop the
            # scaler and selector, so a raw 64-pixel row would no longer predict.
            np.testing.assert_array_equal(
                reloaded.predict(self.X_test), self.pipeline.predict(self.X_test)
            )
            self.assertEqual(Path(model_path).name, "digits_pipeline_lr.pkl")


class ImprovementStudyTests(unittest.TestCase):
    """報告的改善空間討論引用這份研究，數字必須對應真正交出去的管線。"""

    def test_study_baseline_matches_the_saved_pipeline(self):
        import joblib

        study = load_module(ASSIGNMENT_DIR / "07_assignment3_explore.py", "assignment3_explore")
        model = joblib.load(ASSIGNMENT_DIR / "digits_pipeline_lr.pkl")

        baseline = study.baseline_config()

        # 研究的基準組若和實際交出去的管線漂移，報告就會拿不同的模型互相比較。
        self.assertEqual(
            baseline["max_features"],
            int(model.named_steps["selector"].get_support().sum()),
        )
        self.assertEqual(baseline["classifier_C"], model.named_steps["classifier"].C)
        self.assertEqual(baseline["selector_C"], model.named_steps["selector"].estimator.C)

    def test_study_reports_every_group_the_report_section_renders(self):
        study = load_module(ASSIGNMENT_DIR / "07_assignment3_explore.py", "assignment3_explore")

        groups = {spec["group"] for spec in study.EXPERIMENTS}

        self.assertEqual(
            groups,
            {"feature_budget", "selection_method", "hyperparameter", "augmentation", "nonlinear"},
        )


class SubmissionConstraintTests(unittest.TestCase):
    """作業規定：繳交的評估程式不得含任何訓練或模型選取程式碼。"""

    FORBIDDEN = {
        "GridSearchCV",
        "RandomizedSearchCV",
        "StratifiedKFold",
        "cross_val_score",
        "SelectFromModel",
        "StandardScaler",
        "LogisticRegression",
    }

    def test_evaluation_script_imports_no_training_or_model_selection_symbols(self):
        tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)

        self.assertEqual(self.FORBIDDEN & imported, set())

    def test_evaluation_script_never_calls_fit(self):
        tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))

        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        self.assertNotIn("fit", called)
        self.assertNotIn("fit_transform", called)

    def test_evaluation_script_runs_standalone_next_to_only_the_model_file(self):
        with tempfile.TemporaryDirectory() as submission_dir:
            shutil.copy(EVAL_PATH, submission_dir)
            shutil.copy(trained_model_path(), submission_dir)

            completed = subprocess.run(
                [sys.executable, "07_assignment3_eval.py"],
                cwd=submission_dir,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("accuracy", completed.stdout)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = load_eval_module()
        cls.model_path = trained_model_path()

    def test_mask_title_states_how_many_of_the_64_pixels_were_kept(self):
        mask = np.zeros(64, dtype=bool)
        mask[:40] = True

        title = self.evaluation.feature_mask_title(mask)

        self.assertEqual(title, "Feature Selection Mask (Selected: 40/64)")

    def test_loaded_test_set_matches_the_grader_s_360_sample_hold_out(self):
        X_test, y_test = self.evaluation.load_test_set()

        self.assertEqual(X_test.shape, (360, 64))
        np.testing.assert_array_equal(y_test[:10], [6, 9, 3, 7, 2, 1, 5, 2, 5, 2])

    def test_evaluation_accuracy_matches_the_hand_counted_hit_rate(self):
        model = self.evaluation.load_model(self.model_path)
        X_test, y_test = self.evaluation.load_test_set()

        result = self.evaluation.evaluate(model, X_test, y_test)

        hits = int((model.predict(X_test) == y_test).sum())
        self.assertAlmostEqual(result["accuracy"], hits / 360, places=12)
        self.assertGreaterEqual(result["accuracy"], 0.95)

    def test_confusion_matrix_totals_every_test_sample_across_ten_classes(self):
        model = self.evaluation.load_model(self.model_path)
        X_test, y_test = self.evaluation.load_test_set()

        result = self.evaluation.evaluate(model, X_test, y_test)

        self.assertEqual(result["confusion_matrix"].shape, (10, 10))
        self.assertEqual(int(result["confusion_matrix"].sum()), 360)
        # Per-class support is fixed by the pinned split.
        np.testing.assert_array_equal(
            result["confusion_matrix"].sum(axis=1),
            [33, 28, 33, 34, 46, 47, 35, 34, 30, 40],
        )

    def test_run_evaluation_writes_both_required_figures_and_the_text_report(self):
        with tempfile.TemporaryDirectory() as output_dir:
            result = self.evaluation.run_evaluation(
                model_path=self.model_path, output_dir=output_dir
            )

            for filename in (
                "assignment3_feature_mask.png",
                "assignment3_confusion_matrix.png",
                "assignment3_results.txt",
            ):
                self.assertGreater((Path(output_dir) / filename).stat().st_size, 0)

            report = (Path(output_dir) / "assignment3_results.txt").read_text(encoding="utf-8")
            self.assertIn(f"{result['accuracy']:.2f}", report)
            self.assertLessEqual(int(result["feature_mask"].sum()), 44)


if __name__ == "__main__":
    unittest.main()
