# 章節導讀（繁體中文快速指南）

給自己看的學習筆記索引：每章「在做什麼」、關鍵概念、對應要打開的檔案。專業名詞維持英文，方便對照書本原文與日後查資料。

---

## Part 1：古典機器學習 + scikit-learn（ch01–ch10）

### ch01 — 環境設定，沒有程式碼內容
單純是環境檢查章節，跑 `python_environment_check.ipynb` 確認套件版本正確即可，不用花時間讀。

### ch02 — 用最基本的神經元學分類
**在做什麼**：從零手刻兩個最古老的線性分類器——`Perceptron`（感知器）和 `Adaline`（Adaptive Linear Neuron），理解 gradient descent 怎麼一步步更新權重。
**關鍵概念**：perceptron learning rule、cost function、gradient descent、feature scaling、stochastic gradient descent (SGD)
**對應檔案**：`ch02.ipynb`

### ch03 — scikit-learn 分類器總覽
**在做什麼**：把常見分類演算法都跑過一輪，比較彼此的假設與適用場景，是後面章節的地圖。
**關鍵概念**：logistic regression、regularization（避免 overfitting）、SVM（含 kernel trick 處理非線性）、decision tree（information gain）、random forest、k-nearest neighbors
**對應檔案**：`ch03.ipynb`

### ch04 — 資料前處理（實務上最常用的一章）
**在做什麼**：處理真實資料常見的髒亂問題——缺值、類別型欄位、特徵尺度不一致、特徵太多。
**關鍵概念**：missing data 處理、one-hot encoding、train/test split、L1/L2 regularization（特徵選擇）、feature importance（用 random forest 評估)
**對應檔案**：`ch04.ipynb`

### ch05 — 降維：把高維資料壓縮成能看懂的維度
**在做什麼**：三種降維方法比較——PCA 是找資料本身變異量最大的方向；LDA 是利用分類標籤找出最能區分類別的方向；t-SNE 專門用來畫圖視覺化。
**關鍵概念**：PCA（unsupervised）、LDA（supervised）、explained variance、t-SNE
**對應檔案**：`ch05.ipynb`

### ch06 — 模型評估與調參的正確做法
**在做什麼**：這章是「怎麼知道你的模型真的有效」的方法論，很多人跳過但其實最關鍵。
**關鍵概念**：pipeline（串接前處理+模型）、k-fold cross-validation、learning curve / validation curve（判斷 bias vs. variance）、grid search / randomized search（hyperparameter tuning）、confusion matrix、precision/recall、ROC curve、class imbalance 處理
**對應檔案**：`ch06.ipynb`

### ch07 — 集成學習：多個弱模型變強模型
**在做什麼**：三種把多個分類器組合起來的策略。
**關鍵概念**：majority voting、bagging（如 random forest 的原理）、AdaBoost（adaptive boosting）、gradient boosting、XGBoost
**對應檔案**：`ch07.ipynb`

### ch08 — 文字資料應用：情感分析
**在做什麼**：拿 IMDb 影評資料做「這則評論是正評還負評」的分類，也介紹主題模型。
**關鍵概念**：bag-of-words、TF-IDF、text cleaning/tokenization、out-of-core learning（處理超大資料集）、LDA topic modeling（注意這裡的 LDA 跟 ch05 的 LDA 是不同東西，一個是 Latent Dirichlet Allocation 一個是 Linear Discriminant Analysis）
**對應檔案**：`ch08.ipynb`

### ch09 — 迴歸分析：預測連續數值
**在做什麼**：用房價資料集(Ames Housing)示範迴歸的完整流程，從線性到非線性。
**關鍵概念**：linear regression、RANSAC（處理離群值)、regularized regression（Ridge/Lasso/Elastic Net）、polynomial regression、decision tree / random forest regression
**對應檔案**：`ch09.ipynb`

### ch10 — 分群：沒有標籤的資料怎麼找結構
**在做什麼**：三種分群邏輯的比較——k-means 是找圓形群聚中心；hierarchical clustering 是由下而上不斷合併；DBSCAN 是找高密度區域，能處理不規則形狀。
**關鍵概念**：k-means / k-means++、elbow method、silhouette plot、hierarchical clustering（dendrogram）、DBSCAN
**對應檔案**：`ch10.ipynb`

---

## Part 2：深度學習與 PyTorch 基礎（ch11–ch13，全書轉折點）

### ch11 — 手刻多層神經網路（不用任何深度學習框架）
**在做什麼**：全書最重要的一章之一。完全用 numpy 自己寫出 forward propagation 和 backpropagation，親手推導梯度怎麼往回傳。之後所有章節都是這個概念的框架化版本。
**關鍵概念**：multilayer perceptron (MLP)、forward propagation、loss function、backpropagation、MNIST 手寫數字辨識
**對應檔案**：`ch11.ipynb`

### ch12 — 改用 PyTorch 重新做一次
**在做什麼**：把 ch11 手刻的東西換成 PyTorch 的寫法，建立 tensor、DataLoader、`torch.nn`、`torch.optim` 的基本操作直覺。
**關鍵概念**：tensor 操作、`DataLoader`／`Dataset`、`torch.nn.Module`、activation functions（sigmoid / softmax / tanh / ReLU）
**對應檔案**：`ch12.ipynb`

### ch13 — PyTorch 底層機制 + 三個實戰專案
**在做什麼**：深入理解 PyTorch 自動微分怎麼運作，接著用兩個小專案(車輛油耗迴歸、MNIST 分類)練習，最後介紹 PyTorch Lightning 這種更高層的 API。
**關鍵概念**：computation graph、automatic differentiation (`autograd`)、`nn.Sequential` vs. 自訂 `nn.Module`、PyTorch Lightning、TensorBoard
**對應檔案**：`ch13_part1~4.ipynb`（part4 是 ignite，選讀）

---

## Part 3：進階深度學習主題（ch14–ch19，依興趣挑重點）

### ch14 — CNN：影像分類
**在做什麼**：理解卷積運算怎麼抓取影像的局部特徵，並實作人臉微笑辨識(CelebA 資料集)。
**關鍵概念**：convolution、padding、pooling (subsampling)、dropout、data augmentation
**對應檔案**：`ch14_part1.ipynb`, `ch14_part2.ipynb`

### ch15 — RNN：序列資料建模
**在做什麼**：處理「順序有意義」的資料，例如文字。示範兩個專案：影評情感分析、字元級語言模型(自動生成文字)。
**關鍵概念**：RNN 循環機制、long-range dependency 問題、LSTM、embedding layer、bidirectional RNN
**對應檔案**：`ch15_part1.ipynb`, `ch15_part2.ipynb`

### ch16 — Transformer：現代 NLP 的核心架構
**在做什麼**：從 RNN 的 attention 機制講起，一路推到 self-attention、Transformer 架構，最後實際玩 GPT-2 生成文字、微調 BERT 做分類。這章是現在 LLM 技術的基礎，值得花時間。
**關鍵概念**：attention mechanism、self-attention、scaled dot-product attention、multi-head attention、pre-training + fine-tuning、GPT-2、BERT
**對應檔案**：`ch16-part1-self-attention.ipynb`, `ch16-part2-gpt2.ipynb`, `ch16-part3-bert.ipynb`

### ch17 — GAN：生成對抗網路
**在做什麼**：兩個網路互相對抗——generator 想騙過 discriminator，discriminator 想抓出假資料，最後 generator 能生成以假亂真的圖片。
**關鍵概念**：generator / discriminator、autoencoder、DCGAN（convolutional GAN）、WGAN-GP（訓練穩定度改善）、mode collapse
**對應檔案**：`ch17_part1.ipynb`, `ch17_part2.ipynb`

### ch18 — GNN：圖結構資料
**在做什麼**：處理節點與邊構成的圖資料（例如分子結構），示範 graph convolution 怎麼讓每個節點「吸收鄰居的資訊」。
**關鍵概念**：graph representation、graph convolution、message passing、PyTorch Geometric
**對應檔案**：`ch18.ipynb`

### ch19 — 強化學習：從跟環境互動中學習
**在做什麼**：全書壓軸，講 agent 怎麼透過跟環境互動、拿到 reward 來學習策略，從理論(MDP)一路做到 deep Q-learning。
**關鍵概念**：Markov Decision Process (MDP)、policy / value function、dynamic programming、Monte Carlo、TD learning（SARSA / Q-learning）、OpenAI Gym、Deep Q-Network (DQN)
**對應檔案**：`ch19.ipynb`（`gridworld/`、`cartpole/` 是輔助環境程式碼）

---

## 用法建議
- 每次開新章節前先看這份文件對應段落建立心智地圖，再進 notebook
- 「關鍵概念」欄位可以當作自我檢測清單：讀完該章後蓋住答案，看自己能不能對每個名詞講出白話解釋
- ch16、ch19 內容量大，可以拆兩三天讀完，不用一次啃完
