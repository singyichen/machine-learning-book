"""
Assignment #1 - Adaline SGD 分類器

以四項新進人員能力/條件特徵 (Language_Skill, Math_Skill, Coding_Skill,
Health_Condition) 預測員工在職一年後的表現 (Performance_Label: 0 普通 / 1 優秀)。

不使用 scikit-learn。
"""
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AdalineSGD:
    """ADAptive LInear NEuron classifier (Stochastic Gradient Descent)。

    Parameters
    ------------
    eta : float
        初始學習率
    n_iter : int
        訓練輪次 (epochs)
    shuffle : bool
        每個 epoch 開始前是否打散訓練資料
    adaptive : bool
        是否啟用自適應學習率，依 eta / (1 + decay * epoch) 逐輪衰減
    decay : float
        自適應學習率的衰減係數
    random_state : int
        亂數種子，控制初始權重與 shuffle 的重現性
    """

    def __init__(self, eta=0.01, n_iter=15, shuffle=True,
                 adaptive=False, decay=0.01, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.adaptive = adaptive
        self.decay = decay
        self.random_state = random_state
        self.w_initialized = False

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            eta = self.eta / (1 + self.decay * i) if self.adaptive else self.eta
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = [self._update_weights(xi, target, eta) for xi, target in zip(X, y)]
            self.losses_.append(np.mean(losses))
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = 0.0
        self.w_initialized = True

    def _update_weights(self, xi, target, eta):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += eta * 2.0 * xi * error
        self.b_ += eta * 2.0 * error
        return error ** 2

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


class StandardScaler:
    """標準化：平均值/標準差僅用訓練集計算，同一組統計值套用到訓練與測試集。"""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def train_and_evaluate(X_train, y_train, X_test, y_test,
                        eta=0.01, n_iter=80, adaptive=True, random_state=1):
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = AdalineSGD(eta=eta, n_iter=n_iter, shuffle=True,
                        adaptive=adaptive, random_state=random_state)
    model.fit(X_train_std, y_train)

    y_pred = model.predict(X_test_std)
    return model, scaler, accuracy(y_test, y_pred)


def plot_decision_regions(ax, X, y, classifier, feature_names, resolution=0.05):
    markers = ('o', 's')
    point_colors = ('#d62728', '#1f77b4')
    cmap = ListedColormap(('#f7c9c4', '#c9d7f0'))

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)
    ax.contourf(xx1, xx2, lab, alpha=0.6, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], alpha=0.9, c=point_colors[idx],
                   marker=markers[idx], label=f'Class {cl}', edgecolor='black', s=25)

    ax.set_xlabel(f'{feature_names[0]} (standardized)')
    ax.set_ylabel(f'{feature_names[1]} (standardized)')
    ax.set_title(' & '.join(feature_names), fontsize=10)
    ax.legend(loc='upper right', fontsize=7)


def combo_label(combo):
    return ' & '.join(combo) if len(combo) == 2 else 'All Four Features'


def main():
    df = pd.read_csv('tech_employees.csv')

    feature_cols = ['Language_Skill', 'Math_Skill', 'Coding_Skill', 'Health_Condition']
    label_col = 'Performance_Label'

    # 必須依原始順序切分：前 200 筆訓練、後 100 筆測試
    train_df = df.iloc[:200]
    test_df = df.iloc[200:300]

    y_train_full = train_df[label_col].to_numpy()
    y_test_full = test_df[label_col].to_numpy()

    pair_combos = list(itertools.combinations(feature_cols, 2))
    combos = pair_combos + [tuple(feature_cols)]

    results = []
    pair_models = {}

    for combo in combos:
        X_train = train_df[list(combo)].to_numpy(dtype=float)
        X_test = test_df[list(combo)].to_numpy(dtype=float)

        model, scaler, acc = train_and_evaluate(
            X_train, y_train_full, X_test, y_test_full,
            eta=0.01, n_iter=80, adaptive=True, random_state=1,
        )
        results.append({'combo': combo, 'accuracy': acc})
        if len(combo) == 2:
            pair_models[combo] = (model, scaler, X_test, y_test_full)

    results.sort(key=lambda r: r['accuracy'], reverse=True)

    header = f"{'Feature Combination':38s}{'Test Accuracy':>15s}"
    lines = [header, '-' * len(header)]
    for r in results:
        lines.append(f"{combo_label(r['combo']):38s}{r['accuracy']:>15.2f}")

    # 以「包含該特徵之所有組合的平均測試正確率」作為單一特徵鑑別度的排序依據
    feature_avg_acc = {}
    for name in feature_cols:
        accs = [r['accuracy'] for r in results if name in r['combo']]
        feature_avg_acc[name] = np.mean(accs)
    ranked_features = sorted(feature_avg_acc, key=feature_avg_acc.get, reverse=True)
    conclusion = (
        "結論：以「包含該特徵之組合平均正確率」排序，"
        + '、'.join(f"{name} ({feature_avg_acc[name]:.2f})" for name in ranked_features)
        + f"。{ranked_features[0]} 鑑別度最高，{ranked_features[1]} 次之。"
    )

    report = '\n'.join(lines) + '\n\n' + conclusion
    print(report)

    with open('assignment1_results.txt', 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, combo in zip(axes.ravel(), pair_combos):
        model, scaler, X_test, y_test = pair_models[combo]
        X_test_std = scaler.transform(X_test)
        plot_decision_regions(ax, X_test_std, y_test, model, combo)

    plt.tight_layout()
    plt.savefig('assignment1_decision_boundaries.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
