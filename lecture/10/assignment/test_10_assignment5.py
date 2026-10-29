"""Assignment #5 tests."""

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
TRAIN_PATH = ASSIGNMENT_DIR / "10_assignment5_train.py"
EVAL_PATH = ASSIGNMENT_DIR / "10_assignment5_eval.py"


def load_module(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_train_module():
    return load_module(TRAIN_PATH, "assignment5_train")


def load_eval_module():
    return load_module(EVAL_PATH, "assignment5_eval")


_TRAINED_MODEL_DIR = None


def trained_model_path():
    """Train the real pipeline once and reuse the artefact across test classes."""
    global _TRAINED_MODEL_DIR
    if _TRAINED_MODEL_DIR is None:
        _TRAINED_MODEL_DIR = tempfile.mkdtemp(prefix="assignment5_model_")
        train = load_train_module()
        train.run_training(output_dir=_TRAINED_MODEL_DIR)
    return Path(_TRAINED_MODEL_DIR) / "digits_pipeline_lr_pca.pkl"


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


class ComponentBudgetTests(unittest.TestCase):
    """投影片 p.52：PCA 降維後的特徵空間維度必須 <= 40。"""

    def test_validate_component_budget_rejects_41_components(self):
        train = load_train_module()

        with self.assertRaisesRegex(ValueError, "41"):
            train.validate_component_budget(41)

    def test_validate_component_budget_accepts_the_40_component_cap(self):
        train = load_train_module()

        self.assertEqual(train.validate_component_budget(40), 40)

    def test_every_searched_component_count_stays_within_the_cap(self):
        train = load_train_module()

        self.assertTrue(all(n <= 40 for n in train.PARAM_GRID["pca__n_components"]))


class TrainedPipelineTests(unittest.TestCase):
    """The pipeline is trained once and shared, because fitting is slow."""

    @classmethod
    def setUpClass(cls):
        cls.train = load_train_module()
        X, y = cls.train.load_digit_images()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = cls.train.split_train_test(X, y)
        cls.search = cls.train.select_best_model(cls.X_train, cls.y_train)
        cls.pipeline = cls.search.best_estimator_

    def test_pipeline_extracts_at_most_40_components_from_the_64_pixels(self):
        n_components = self.train.extract_n_components(self.pipeline)

        self.assertLessEqual(n_components, 40)
        self.assertEqual(
            self.pipeline.named_steps["pca"].components_.shape, (n_components, 64)
        )

    def test_classifier_is_fitted_on_the_extracted_components_not_raw_pixels(self):
        n_components = self.train.extract_n_components(self.pipeline)

        classifier = self.pipeline.named_steps["classifier"]

        # 若分類器接到的是原始 64 維，代表 PCA 沒有真的接進管線。
        self.assertEqual(classifier.coef_.shape[1], n_components)
        self.assertLess(classifier.coef_.shape[1], 64)

    def test_pipeline_scales_features_before_pca(self):
        # PCA 對尺度敏感，特徵縮放必須排在 PCA 之前。
        self.assertEqual(
            [name for name, _ in self.pipeline.steps], ["scaler", "pca", "classifier"]
        )

    def test_test_set_accuracy_clears_the_95_percent_requirement(self):
        accuracy = (self.pipeline.predict(self.X_test) == self.y_test).mean()

        self.assertGreaterEqual(accuracy, 0.95)

    def test_saved_model_reloads_and_predicts_identically(self):
        import joblib

        with tempfile.TemporaryDirectory() as output_dir:
            model_path = self.train.save_pipeline(self.pipeline, output_dir)
            reloaded = joblib.load(model_path)

            # 只 dump 分類器會掉了 scaler 與 PCA，原始 64 維像素就再也預測不了。
            np.testing.assert_array_equal(
                reloaded.predict(self.X_test), self.pipeline.predict(self.X_test)
            )
            self.assertEqual(Path(model_path).name, "digits_pipeline_lr_pca.pkl")


class ExplainedVarianceTests(unittest.TestCase):
    """投影片 p.54 的預期結果 1：維度與總解釋變異量。"""

    @classmethod
    def setUpClass(cls):
        cls.evaluation = load_eval_module()
        import joblib

        cls.model = joblib.load(trained_model_path())

    def test_total_explained_variance_sums_the_pca_ratios(self):
        pca = self.model.named_steps["pca"]

        total = self.evaluation.total_explained_variance(self.model)

        self.assertAlmostEqual(total, float(pca.explained_variance_ratio_.sum()), places=12)
        self.assertGreater(total, 0.9)
        self.assertLessEqual(total, 1.0)

    def test_dimension_line_states_both_the_new_and_original_dimensions(self):
        line = self.evaluation.dimension_line(40)

        self.assertEqual(
            line, "PCA 提取後的特徵維度 (n_components): 40 (從原本的 64 維降至 40 維)"
        )

    def test_explained_variance_line_is_a_percentage_with_two_decimals(self):
        line = self.evaluation.explained_variance_line(0.951843)

        self.assertEqual(line, "特徵提取的總解釋變異量: 95.18%")


class SubmissionConstraintTests(unittest.TestCase):
    """作業規定：繳交的評估程式不得含任何訓練或模型選取程式碼。"""

    FORBIDDEN = {
        "GridSearchCV",
        "RandomizedSearchCV",
        "StratifiedKFold",
        "cross_val_score",
        "PCA",
        "StandardScaler",
        "LogisticRegression",
    }

    def test_evaluation_script_imports_no_training_or_model_selection_symbols(self):
        tree = ast.parse(EVAL_PATH.read_text(encoding="utf-8"))

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
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
                [sys.executable, "10_assignment5_eval.py"],
                cwd=submission_dir,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("accuracy", completed.stdout)
            self.assertIn("PCA 提取後的特徵維度", completed.stdout)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = load_eval_module()
        cls.model_path = trained_model_path()

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
                "assignment5_confusion_matrix.png",
                "assignment5_misclassified.png",
                "assignment5_results.txt",
            ):
                self.assertGreater((Path(output_dir) / filename).stat().st_size, 0)

            report = (Path(output_dir) / "assignment5_results.txt").read_text(encoding="utf-8")
            self.assertIn(f"{result['accuracy']:.2f}", report)
            self.assertIn(
                self.evaluation.misclassified_message(len(result["misclassified"])), report
            )
            self.assertIn(self.evaluation.dimension_line(result["n_components"]), report)


class ExpectedFigureStyleTests(unittest.TestCase):
    """投影片 p.55–56 的預期結果圖要一對一、同樣式，助教會逐張比對。"""

    @classmethod
    def setUpClass(cls):
        cls.evaluation = load_eval_module()
        model = cls.evaluation.load_model(trained_model_path())
        cls.X_test, cls.y_test = cls.evaluation.load_test_set()
        cls.result = cls.evaluation.evaluate(model, cls.X_test, cls.y_test)

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_confusion_figure_labels_all_100_cells_including_zeros_with_no_title(self):
        matrix = self.result["confusion_matrix"]

        fig = self.evaluation.make_confusion_matrix_figure(matrix)
        ax = fig.axes[0]

        self.assertEqual(ax.get_title(), "")
        self.assertEqual(ax.get_xlabel(), "Predicted label")
        self.assertEqual(ax.get_ylabel(), "True label")
        self.assertEqual(sorted(int(t.get_text()) for t in ax.texts), sorted(matrix.ravel()))

    def test_misclassified_indices_are_exactly_the_wrong_predictions(self):
        y_pred = self.result["y_pred"]

        indices = self.evaluation.find_misclassified(self.y_test, y_pred)

        np.testing.assert_array_equal(indices, np.flatnonzero(y_pred != self.y_test))
        np.testing.assert_array_equal(self.result["misclassified"], indices)

    def test_misclassified_count_matches_the_confusion_matrix_off_diagonal(self):
        matrix = self.result["confusion_matrix"]

        off_diagonal = int(matrix.sum() - np.trace(matrix))

        self.assertEqual(len(self.result["misclassified"]), off_diagonal)
        self.assertEqual(
            self.evaluation.misclassified_message(off_diagonal),
            f"測試集總共誤判了 {off_diagonal} 張圖片。",
        )

    def test_misclassified_figure_titles_true_label_and_marks_prediction_in_red(self):
        from matplotlib.colors import to_rgba

        indices = self.result["misclassified"]
        y_pred = self.result["y_pred"]

        fig = self.evaluation.make_misclassified_figure(self.X_test, self.y_test, y_pred, indices)
        shown = [ax for ax in fig.axes if ax.images]

        self.assertEqual(len(shown), len(indices))
        for ax, index in zip(shown, indices):
            self.assertEqual(ax.get_title(), f"True: {self.y_test[index]}")
            self.assertEqual(ax.images[0].get_array().shape, (8, 8))
            [label] = ax.texts
            self.assertEqual(label.get_text(), str(y_pred[index]))
            self.assertEqual(to_rgba(label.get_color()), to_rgba("red"))
        # 投影片為每列 5 張；誤判超過 5 張就要換列。
        rows, columns = shown[0].get_subplotspec().get_gridspec().get_geometry()
        self.assertEqual(columns, 5)
        self.assertEqual(rows, -(-len(indices) // 5))


class ImprovementStudyTests(unittest.TestCase):
    """報告的改善空間討論引用這份研究，數字必須對應真正交出去的管線。"""

    def test_study_baseline_matches_the_saved_pipeline(self):
        import joblib

        study = load_module(ASSIGNMENT_DIR / "10_assignment5_explore.py", "assignment5_explore")
        model = joblib.load(ASSIGNMENT_DIR / "digits_pipeline_lr_pca.pkl")

        baseline = study.baseline_config()

        # 研究的基準組若和實際交出去的管線漂移，報告就會拿不同的模型互相比較。
        self.assertEqual(baseline["n_components"], model.named_steps["pca"].n_components_)
        self.assertEqual(baseline["classifier_C"], model.named_steps["classifier"].C)

    def test_study_reports_every_group_the_report_section_renders(self):
        study = load_module(ASSIGNMENT_DIR / "10_assignment5_explore.py", "assignment5_explore")

        groups = {spec["group"] for spec in study.EXPERIMENTS}

        self.assertEqual(
            groups,
            {"component_budget", "whitening", "hyperparameter", "extraction_method", "nonlinear"},
        )


if __name__ == "__main__":
    unittest.main()
