"""Assignment #6 tests."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ASSIGNMENT_DIR = Path(__file__).resolve().parent
MODULE_PATH = ASSIGNMENT_DIR / "11_assignment6.py"


def load_module(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_assignment():
    return load_module(MODULE_PATH, "assignment6")


_RESULTS = None


def shared_results():
    """三個模型只訓練一次，跨測試類別共用（MLP 訓練較慢）。"""
    global _RESULTS
    if _RESULTS is None:
        module = load_assignment()
        _RESULTS = module.run_all_models()
    return _RESULTS


class DatasetContractTests(unittest.TestCase):
    """投影片 p.41：442 筆、10 維、target 介於 25 到 346。"""

    def test_dataset_is_442_patients_with_10_clinical_features(self):
        module = load_assignment()

        X, y = module.load_diabetes_data()

        self.assertEqual(X.shape, (442, 10))
        self.assertEqual(y.shape, (442,))

    def test_target_is_the_quantitative_progression_measure_between_25_and_346(self):
        module = load_assignment()

        _, y = module.load_diabetes_data()

        self.assertEqual((float(y.min()), float(y.max())), (25.0, 346.0))

    def test_features_are_pre_scaled_to_unit_sum_of_squares(self):
        module = load_assignment()

        X, _ = module.load_diabetes_data()

        # 投影片 p.41 註記：各項特徵已預先經過縮放（每項特徵的平方和皆為 1）。
        np.testing.assert_allclose((X ** 2).sum(axis=0), np.ones(10), atol=1e-8)


class SplitContractTests(unittest.TestCase):
    """投影片 p.42：80% 訓練、20% 測試；p.43 註明測試集約 89 筆。"""

    def test_split_holds_out_89_of_the_442_samples(self):
        module = load_assignment()

        X_train, X_test, y_train, y_test = module.split_train_test(*module.load_diabetes_data())

        self.assertEqual(X_train.shape, (353, 10))
        self.assertEqual(X_test.shape, (89, 10))
        self.assertEqual(y_train.shape, (353,))
        self.assertEqual(y_test.shape, (89,))

    def test_split_is_pinned_to_seed_42(self):
        module = load_assignment()

        _, _, _, y_test = module.split_train_test(*module.load_diabetes_data())

        # Hand-derived from train_test_split(..., test_size=0.2, random_state=42).
        np.testing.assert_array_equal(y_test[:5], [219.0, 70.0, 202.0, 230.0, 111.0])


class MetricTests(unittest.TestCase):
    def test_evaluate_returns_mse_and_r2_matching_hand_computation(self):
        module = load_assignment()
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 39.0])

        metrics = module.evaluate_predictions(y_true, y_pred)

        expected_mse = float(np.mean((y_true - y_pred) ** 2))
        expected_r2 = 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
        self.assertAlmostEqual(metrics["mse"], expected_mse, places=10)
        self.assertAlmostEqual(metrics["r2"], expected_r2, places=10)


class LinearBaselineTests(unittest.TestCase):
    """投影片 p.43 把多元線性回歸當作效能 baseline，數值可直接核對。"""

    def test_linear_regression_reproduces_the_slide_baseline(self):
        results = shared_results()

        linear = results["Linear Regression"]

        # 投影片 p.43: MSE 2900.19 | R2 Score 0.4526
        self.assertAlmostEqual(linear["mse"], 2900.19, delta=1.0)
        self.assertAlmostEqual(linear["r2"], 0.4526, places=3)


class PolynomialTests(unittest.TestCase):
    """投影片 p.42 指定 PolynomialFeatures degree=2。"""

    def test_polynomial_model_expands_the_10_features_to_degree_two_terms(self):
        module = load_assignment()

        model = module.build_polynomial_model()
        X_train, _, y_train, _ = module.split_train_test(*module.load_diabetes_data())
        model.fit(X_train, y_train)

        poly = model.named_steps["poly"]
        self.assertEqual(poly.degree, 2)
        # 10 個原始特徵 + 10 個平方項 + 45 個交互作用項 = 65 維
        self.assertEqual(poly.transform(X_train).shape[1], 65)

    def test_polynomial_r2_lands_in_the_range_the_slides_predict(self):
        results = shared_results()

        # 投影片 p.45：多項式回歸 (degree=2) 的 R2 區間為 0.30 ~ 0.42（或負值）。
        self.assertLessEqual(results["Polynomial Regression"]["r2"], 0.42)


class MLPTests(unittest.TestCase):
    """投影片 p.42 要求以 PyTorch 實作 MLP regressor。"""

    def test_mlp_is_a_small_pytorch_network_as_the_slides_require(self):
        import torch

        module = load_assignment()

        model = module.build_mlp(n_features=10)

        self.assertIsInstance(model, torch.nn.Module)
        # 投影片 p.45：架構必須非常小，否則必過擬合。
        n_parameters = sum(p.numel() for p in model.parameters())
        self.assertLess(n_parameters, 10000)

    def test_build_mlp_respects_the_requested_hidden_sizes(self):
        import torch

        module = load_assignment()

        small = module.build_mlp(10, hidden_sizes=(8,))
        large = module.build_mlp(10, hidden_sizes=(128, 64))

        widths = lambda model: [  # noqa: E731
            layer.out_features for layer in model if isinstance(layer, torch.nn.Linear)
        ]
        self.assertEqual(widths(small), [8, 1])
        self.assertEqual(widths(large), [128, 64, 1])

    def test_train_mlp_passes_the_requested_hidden_sizes_through(self):
        import torch

        module = load_assignment()
        X_train, _, y_train, _ = module.split_train_test(*module.load_diabetes_data())

        history = module.train_mlp(X_train, y_train, hidden_sizes=(8,))

        # 改善空間研究靠這條路徑比較不同容量；若參數沒傳下去，四組會得到同一個數字。
        widths = [
            layer.out_features
            for layer in history["model"]
            if isinstance(layer, torch.nn.Linear)
        ]
        self.assertEqual(widths, [8, 1])

    def test_mlp_never_sees_the_test_set_during_training(self):
        module = load_assignment()
        X_train, X_test, y_train, y_test = module.split_train_test(*module.load_diabetes_data())

        history = module.train_mlp(X_train, y_train)

        # early stopping 用的驗證集必須從訓練集切出來，測試集只能做最後一次評估。
        self.assertEqual(history["n_train"] + history["n_valid"], len(y_train))
        self.assertGreater(history["n_valid"], 0)

    def test_mlp_stops_early_instead_of_running_every_epoch(self):
        module = load_assignment()
        X_train, _, y_train, _ = module.split_train_test(*module.load_diabetes_data())

        history = module.train_mlp(X_train, y_train)

        # 投影片 p.45：必須提早停止 (Early Stopping) 否則必過擬合。
        # 要檢查迴圈真的中斷了——「最佳輪次不是最後一輪」即使跑滿 MAX_EPOCHS 也會成立，
        # 拿它當斷言會讓「把 break 拿掉」這個改壞完全驗不出來。
        self.assertLess(len(history["valid_losses"]), module.MAX_EPOCHS)
        self.assertEqual(
            len(history["valid_losses"]), history["best_epoch"] + module.PATIENCE
        )
        self.assertEqual(
            history["best_valid_loss"], min(history["valid_losses"])
        )

    def test_mlp_training_is_reproducible_under_the_pinned_seed(self):
        module = load_assignment()
        X_train, X_test, y_train, _ = module.split_train_test(*module.load_diabetes_data())

        first = module.train_mlp(X_train, y_train)
        second = module.train_mlp(X_train, y_train)

        np.testing.assert_allclose(
            first["predict"](X_test), second["predict"](X_test), rtol=1e-6
        )


class RequirementTests(unittest.TestCase):
    """投影片 p.43：三者之中最佳的 R2 分數必須達 0.46 以上。"""

    def test_all_three_required_models_are_reported(self):
        results = shared_results()

        self.assertEqual(
            list(results),
            ["Linear Regression", "Polynomial Regression", "PyTorch MLP Regressor"],
        )

    def test_best_r2_clears_the_0_46_requirement(self):
        results = shared_results()

        best = max(row["r2"] for row in results.values())

        self.assertGreaterEqual(best, 0.46)

    def test_every_model_is_scored_on_the_same_89_sample_test_set(self):
        results = shared_results()

        for name, row in results.items():
            with self.subTest(model=name):
                self.assertEqual(len(row["y_pred"]), 89)


class ComparisonTableTests(unittest.TestCase):
    """投影片 p.43 的預期結果 1：量化指標比較表的字樣與格式。"""

    def test_table_matches_the_slide_layout_line_by_line(self):
        module = load_assignment()
        results = {
            "Linear Regression": {"mse": 2900.19, "r2": 0.4526},
            "Polynomial Regression": {"mse": 3096.03, "r2": 0.4156},
            "PyTorch MLP Regressor": {"mse": 2586.11, "r2": 0.5119},
        }

        table = module.format_comparison_table(results)

        self.assertEqual(
            table.splitlines(),
            [
                "========== 各模型效能比較 ==========",
                "[Linear Regression] MSE: 2900.19 | R² Score: 0.4526",
                "[Polynomial Regression] MSE: 3096.03 | R² Score: 0.4156",
                "[PyTorch MLP Regressor] MSE: 2586.11 | R² Score: 0.5119",
            ],
        )


class BarChartStyleTests(unittest.TestCase):
    """投影片 p.44 的預期結果 2：長條圖要一對一、同樣式。"""

    @classmethod
    def setUpClass(cls):
        cls.module = load_assignment()
        cls.results = shared_results()

    def tearDown(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_chart_title_states_it_is_the_test_set(self):
        fig = self.module.make_comparison_figure(self.results)
        ax = fig.axes[0]

        self.assertEqual(ax.get_title(), "Comparison of Regression Models (Test Set)")
        self.assertEqual(ax.get_ylabel(), "R² Score")

    def test_chart_draws_one_bar_per_model_at_its_r2_height(self):
        fig = self.module.make_comparison_figure(self.results)
        ax = fig.axes[0]

        bars = ax.containers[0]
        self.assertEqual(len(bars), 3)
        for bar, name in zip(bars, self.results):
            self.assertAlmostEqual(bar.get_height(), self.results[name]["r2"], places=10)
        self.assertEqual(
            [label.get_text() for label in ax.get_xticklabels()], list(self.results)
        )

    def test_each_bar_is_annotated_with_its_r2_to_four_decimals(self):
        fig = self.module.make_comparison_figure(self.results)
        ax = fig.axes[0]

        annotations = sorted(text.get_text() for text in ax.texts)

        self.assertEqual(
            annotations, sorted(f"{row['r2']:.4f}" for row in self.results.values())
        )

    def test_y_axis_spans_the_negative_to_one_range_the_slide_shows(self):
        fig = self.module.make_comparison_figure(self.results)
        ax = fig.axes[0]

        bottom, top = ax.get_ylim()
        self.assertAlmostEqual(bottom, -0.2, places=10)
        self.assertAlmostEqual(top, 1.0, places=10)


class ScriptTests(unittest.TestCase):
    def test_script_runs_standalone_and_prints_the_comparison_table(self):
        with tempfile.TemporaryDirectory() as run_dir:
            shutil.copy(MODULE_PATH, run_dir)

            completed = subprocess.run(
                [sys.executable, "11_assignment6.py"],
                cwd=run_dir,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("========== 各模型效能比較 ==========", completed.stdout)
            self.assertIn("[PyTorch MLP Regressor]", completed.stdout)
            for filename in ("assignment6_model_comparison.png", "assignment6_results.txt"):
                self.assertGreater((Path(run_dir) / filename).stat().st_size, 0)


class ImprovementStudyTests(unittest.TestCase):
    """報告的改善空間討論引用這份研究。"""

    def test_study_reports_every_group_the_report_section_renders(self):
        study = load_module(ASSIGNMENT_DIR / "11_assignment6_explore.py", "assignment6_explore")

        groups = {spec["group"] for spec in study.EXPERIMENTS}

        self.assertEqual(
            groups,
            {"regularization", "polynomial_fix", "mlp_capacity", "other_algorithms"},
        )


if __name__ == "__main__":
    unittest.main()
