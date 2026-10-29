"""Assignment #4 tests.

繳交腳本的架構是手抄進 model_X_test.py 的，與訓練時的定義很容易漂移，
所以 SubmissionScriptTests 會實際執行繳交腳本一次：架構只要有一層對不上，
load_state_dict() 就會直接失敗。這組測試較慢（方法二需在 10,000 張 224×224
影像上推論），屬於正常現象。
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch


ASSIGNMENT_DIR = Path(__file__).resolve().parent
CELEBA_ROOT = ASSIGNMENT_DIR.parent / "CelebA"
MODEL_1_TEST = ASSIGNMENT_DIR / "model_1_test.py"
MODEL_2_TEST = ASSIGNMENT_DIR / "model_2_test.py"


def load_module(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_common():
    return load_module(ASSIGNMENT_DIR / "09_assignment4_common.py", "assignment4_common")


def celeba_available():
    return (CELEBA_ROOT / "celeba" / "img_align_celeba").is_dir()


class DatasetContractTests(unittest.TestCase):
    """作業規定：兩種方法的資料集條件須一致。"""

    def test_both_methods_use_the_first_16000_training_samples(self):
        common = load_common()

        self.assertEqual(common.N_TRAIN, 16000)

    def test_both_methods_use_the_first_10000_test_samples(self):
        common = load_common()

        self.assertEqual(common.N_TEST, 10000)

    def test_smiling_is_celeba_attribute_index_31(self):
        common = load_common()

        attributes = torch.arange(40)

        # CelebA 的 40 維屬性中，Smiling 位於索引 31（課堂範例 08-3.py 亦同）。
        self.assertEqual(common.SMILING_ATTR_INDEX, 31)
        self.assertEqual(int(common.get_smile(attributes)), 31)

    @unittest.skipUnless(celeba_available(), "本機尚未準備 CelebA 資料集")
    def test_subsets_are_the_required_sizes_and_start_at_the_split_boundary(self):
        common = load_common()

        train, valid, test = common.load_datasets(64, normalize=False)

        self.assertEqual((len(train), len(valid), len(test)), (16000, 1000, 10000))
        # 測試集的第一筆必須是官方 test split 的第一張影像。
        self.assertEqual(test.dataset.filename[0], "182638.jpg")
        self.assertEqual(train.dataset.filename[0], "000001.jpg")


class TransformTests(unittest.TestCase):
    def test_training_transform_augments_but_evaluation_transform_does_not(self):
        common = load_common()

        transform_train, transform_eval = common.build_transforms(64, normalize=False)

        names = lambda compose: [type(t).__name__ for t in compose.transforms]  # noqa: E731
        self.assertIn("RandomCrop", names(transform_train))
        self.assertIn("RandomHorizontalFlip", names(transform_train))
        # 評估不可有隨機性，否則同一個模型每次跑分數都不同。
        self.assertNotIn("RandomCrop", names(transform_eval))
        self.assertNotIn("RandomHorizontalFlip", names(transform_eval))
        self.assertIn("CenterCrop", names(transform_eval))

    def test_only_the_resnet_path_applies_imagenet_normalization(self):
        common = load_common()

        _, without = common.build_transforms(64, normalize=False)
        _, with_normalize = common.build_transforms(224, normalize=True)

        names = lambda compose: [type(t).__name__ for t in compose.transforms]  # noqa: E731
        self.assertNotIn("Normalize", names(without))
        self.assertIn("Normalize", names(with_normalize))

    def test_transforms_produce_the_input_size_each_method_expects(self):
        common = load_common()
        image = torch.randint(0, 255, (3, 218, 178), dtype=torch.uint8)

        _, scratch = common.build_transforms(common.IMAGE_SIZE_SCRATCH, normalize=False)
        _, resnet = common.build_transforms(common.IMAGE_SIZE_RESNET, normalize=True)

        self.assertEqual(tuple(scratch(image).shape), (3, 64, 64))
        self.assertEqual(tuple(resnet(image).shape), (3, 224, 224))


class Method1ArchitectureTests(unittest.TestCase):
    """方法一：調整網路架構（投影片 p.67 列出的三種手段）。"""

    def test_model_adds_batch_normalisation_after_every_convolution(self):
        import torch.nn as nn

        common = load_common()

        model = common.build_model_1()

        convolutions = [m for m in model if isinstance(m, nn.Conv2d)]
        batch_norms = [m for m in model if isinstance(m, nn.BatchNorm2d)]
        self.assertEqual(len(convolutions), 4)
        self.assertEqual(len(batch_norms), 4)
        self.assertEqual(
            [b.num_features for b in batch_norms], [c.out_channels for c in convolutions]
        )

    def test_convolutional_dropout_is_lighter_than_the_sample_code(self):
        import torch.nn as nn

        common = load_common()

        model = common.build_model_1()

        rates = [m.p for m in model if isinstance(m, nn.Dropout)]
        # 課堂範例在卷積層用 0.5，對特徵圖而言等於丟掉一半通道。
        self.assertTrue(all(rate < 0.5 for rate in rates), rates)

    def test_model_outputs_one_probability_per_sample(self):
        common = load_common()
        batch = torch.rand(2, 3, 64, 64)

        output = common.build_model_1()(batch)

        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertTrue(bool(((output >= 0) & (output <= 1)).all()))


class Method2ArchitectureTests(unittest.TestCase):
    """方法二：torchvision 預訓練 ResNet-18 微調（投影片 p.67）。"""

    def test_backbone_is_resnet18_with_pretrained_weights_loaded(self):
        common = load_common()

        pretrained = common.build_model_2()
        random = common.build_model_2(weights=None)

        # 若忘了帶 weights，骨幹就只是隨機初始化，微調的意義完全消失。
        self.assertFalse(
            torch.equal(pretrained.conv1.weight, random.conv1.weight),
            "ResNet-18 載入的似乎不是預訓練權重",
        )

    def test_output_layer_is_replaced_for_binary_classification(self):
        import torch.nn as nn

        common = load_common()

        model = common.build_model_2(weights=None)

        self.assertIsInstance(model.fc, nn.Sequential)
        linear = model.fc[0]
        self.assertEqual(linear.in_features, 512)
        self.assertEqual(linear.out_features, 1)
        self.assertIsInstance(model.fc[1], nn.Sigmoid)

    def test_model_outputs_one_probability_per_sample(self):
        common = load_common()
        batch = torch.rand(2, 3, 224, 224)

        model = common.build_model_2(weights=None).eval()
        with torch.no_grad():
            output = model(batch)

        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertTrue(bool(((output >= 0) & (output <= 1)).all()))


class SavedWeightTests(unittest.TestCase):
    """存檔的權重必須對得上訓練時的架構。"""

    def _assert_state_dict_matches(self, model, weight_path):
        self.assertTrue(weight_path.exists(), f"{weight_path.name} 尚未訓練產生")
        saved = torch.load(weight_path, weights_only=True, map_location="cpu")
        expected = model.state_dict()

        self.assertEqual(set(saved), set(expected))
        for key in expected:
            self.assertEqual(saved[key].shape, expected[key].shape, key)

    def test_model_1_weights_match_the_method_1_architecture(self):
        common = load_common()

        self._assert_state_dict_matches(common.build_model_1(), ASSIGNMENT_DIR / "model_1.pth")

    def test_model_2_weights_match_the_method_2_architecture(self):
        common = load_common()

        self._assert_state_dict_matches(
            common.build_model_2(weights=None), ASSIGNMENT_DIR / "model_2.pth"
        )


class SubmissionFormTests(unittest.TestCase):
    """繳交的是四個檔案打包的單一 zip，測試腳本必須自給自足。"""

    def test_submission_scripts_do_not_import_the_unsubmitted_helpers(self):
        for path in (MODEL_1_TEST, MODEL_2_TEST):
            with self.subTest(script=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                imported = set()
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        imported.update(alias.name for alias in node.names)
                        if isinstance(node, ast.ImportFrom) and node.module:
                            imported.add(node.module)

                # zip 裡只有 model_X.pth 與 model_X_test.py，沒有共用模組。
                for forbidden in ("assignment4_common", "celeba_local_patch",
                                  "prepare_celeba_local"):
                    self.assertNotIn(forbidden, imported)

    def test_submission_scripts_never_train(self):
        for path in (MODEL_1_TEST, MODEL_2_TEST):
            with self.subTest(script=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                called = {
                    node.func.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                }
                self.assertNotIn("backward", called)
                self.assertNotIn("step", called)

    def test_submission_scripts_load_their_own_weight_file(self):
        for path, weights in ((MODEL_1_TEST, "model_1.pth"), (MODEL_2_TEST, "model_2.pth")):
            with self.subTest(script=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn(f'torch.load("{weights}"', source)

    def test_submission_scripts_evaluate_the_first_10000_test_samples(self):
        for path in (MODEL_1_TEST, MODEL_2_TEST):
            with self.subTest(script=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn("number_of_test_samples = 10000", source)
                self.assertIn('split="test"', source)


@unittest.skipUnless(celeba_available(), "本機尚未準備 CelebA 資料集")
class SubmissionScriptTests(unittest.TestCase):
    """實際執行繳交腳本：架構若與權重不符，load_state_dict() 會直接失敗。"""

    ACCURACY_PATTERN = re.compile(r"accuracy\s+([0-9.]+)\s+10000")
    MISCLASSIFIED_PATTERN = re.compile(r"Total misclassified samples: (\d+) / 10000")

    def _run(self, script_name, weight_name):
        with tempfile.TemporaryDirectory() as submission_dir:
            shutil.copy(ASSIGNMENT_DIR / script_name, submission_dir)
            shutil.copy(ASSIGNMENT_DIR / weight_name, submission_dir)
            # 繳交腳本用相對路徑 "CelebA"，助教環境的資料集就放在腳本旁邊。
            os.symlink(CELEBA_ROOT, Path(submission_dir) / "CelebA")

            environment = {**os.environ, "MPLBACKEND": "Agg",
                           "PYTHONPATH": str(ASSIGNMENT_DIR)}
            completed = subprocess.run(
                [sys.executable, "-c",
                 "import celeba_local_patch, runpy, sys; "
                 "celeba_local_patch.apply(sys.argv[1]); "
                 f"runpy.run_path('{script_name}', run_name='__main__')",
                 str(CELEBA_ROOT)],
                cwd=submission_dir,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-3000:])
            return completed.stdout

    def _assert_meets_requirements(self, output):
        self.assertIn("詳細分類報告 (Classification Report):", output)
        self.assertIn("Not Smiling", output)
        self.assertIn("Smiling", output)

        accuracy = self.ACCURACY_PATTERN.search(output)
        self.assertIsNotNone(accuracy, output[-2000:])
        # 作業要求：改善 CelebA 微笑分類任務的性能（正確率 > 90%）
        self.assertGreater(float(accuracy.group(1)), 0.90)

        misclassified = self.MISCLASSIFIED_PATTERN.search(output)
        self.assertIsNotNone(misclassified, output[-2000:])
        # 誤判總數必須與分類報告的正確率一致。
        self.assertAlmostEqual(
            1 - int(misclassified.group(1)) / 10000, float(accuracy.group(1)), places=2
        )

    def test_method_1_script_runs_and_clears_the_90_percent_requirement(self):
        self._assert_meets_requirements(self._run("model_1_test.py", "model_1.pth"))

    def test_method_2_script_runs_and_clears_the_90_percent_requirement(self):
        self._assert_meets_requirements(self._run("model_2_test.py", "model_2.pth"))


if __name__ == "__main__":
    unittest.main()
