# Video Game Data Mining & Machine Learning Pipeline

An end-to-end **Data Mining and Machine Learning** pipeline built in Python and applied to a dataset of **7,001 video games** (`7000videogame.csv`). The project encompasses exploratory data analysis (EDA), multi-modal feature engineering (numerical, categorical, and NLP text reviews), supervised learning (custom Gradient Descent optimizers and ensemble regressors), binary classification, unsupervised clustering with Convex Hull visualization, market-basket association rule mining, and a content-based game recommendation engine.

---

## Key Modules & Capabilities

1. **Data Preprocessing & Exploratory Data Analysis (`preprocessing.py`):**
   * **Dataset Overview:** Analyzes 7,001 records across 18 raw attributes (`Release Year` 2010–2023, `Game Length Hours` 5.0–60.0, `Min Number of Players` 1–10) and transforms them into a **50-feature** encoded matrix.
   * **EDA Visualizations:** Generates distribution histograms, outlier boxplots, top-category bar charts, before/after scaling comparisons, numerical correlation heatmaps, and top TF-IDF terms.
   * **Feature Engineering:** Combines **One-Hot Encoding** (low-cardinality attributes), **Frequency Encoding** (`Publisher`, `Developer`), **TF-IDF Vectorization** (`User Review Text`), and **StandardScaler** normalization.

2. **Supervised Learning: Regression & Classification (`regression_and_classification.py`):**
   * **From-Scratch Linear Regression:** Implements **Batch Gradient Descent**, **Stochastic Gradient Descent (SGD)**, and **Mini-Batch Gradient Descent** to predict `User Rating`, benchmarked against **Ridge Regression**, **Random Forest**, and **Gradient Boosting**.
   * **Dimensionality Reduction (PCA):** Evaluates model performance before and after retaining **95% variance** via PCA.
   * **Binary Classification:** Custom **Logistic Regression** predicting above-average vs. below-average user ratings, evaluated via Confusion Matrix, ROC-AUC, and Precision-Recall curves.

3. **Unsupervised Clustering (`clustering.py`):**
   * Determines the optimal cluster count ($k = 3$) using the **Elbow Method (Inertia)** and **Silhouette Score** (`0.1656`).
   * Visualizes cluster convergence iteratively in 2D space with **Convex Hull** boundaries and extracts mean feature profiles across clusters.

4. **Association Rules & Recommender System (`association_and_recommender.py`):**
   * **Market-Basket Analysis:** Extracts frequent itemsets (`min_support = 0.01`) and high-confidence association rules via the **Apriori** algorithm (`mlxtend`).
   * **Content-Based Recommender:** Computes **Cosine Similarity** across standardized metadata, categorical encodings, and TF-IDF review vectors to recommend similar titles.

---

## Experimental Results

### 1. Regression Performance (Predicting `User Rating`)

| Model | Setting | Test MSE | Test $R^2$ |
| :--- | :--- | :---: | :---: |
| **Mini-Batch Gradient Descent** | Without PCA | **35.2701** | **0.3987** |
| **Random Forest Regressor** | Without PCA | 35.7784 | 0.3901 |
| **Gradient Boosting Regressor** | Without PCA | 36.5782 | 0.3764 |
| **Batch Gradient Descent** | Without PCA | 50.9978 | 0.1306 |
| **Stochastic Gradient Descent (SGD)** | Without PCA | 57.7298 | 0.0158 |
| **Mini-Batch Gradient Descent** | With PCA (95% Var.) | 35.7242 | 0.3910 |
| **Stochastic Gradient Descent (SGD)** | With PCA (95% Var.) | 46.2791 | 0.2110 |
| **Batch Gradient Descent** | With PCA (95% Var.) | 50.9152 | 0.1320 |

### 2. Binary Classification Performance (`Logistic Regression`)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Below Average (0)** | 0.71 | 0.70 | 0.70 | 1,049 |
| **Above Average (1)** | 0.70 | 0.71 | 0.71 | 1,052 |
| **Overall Accuracy / Weighted Avg** | **0.71** | **0.71** | **0.71** | **2,101** |

### 3. Unsupervised Clustering (`K-Means`, Optimal $k = 3$)

| Cluster | Game Count | Avg. Price ($) | Avg. Release Year | Avg. Min Players |
| :--- | :---: | :---: | :---: | :---: |
| **Cluster 0** | 2,331 | $39.76 | 2016.6 | 5.1 |
| **Cluster 1** | 2,365 | $39.64 | 2016.5 | 5.1 |
| **Cluster 2** | 2,305 | $39.43 | 2016.4 | 5.2 |

### 4. Sample Content-Based Recommendations (Query: *Grand Theft Auto V*)

| Rank | Recommended Game | Cosine Similarity |
| :---: | :--- | :---: |
| 1 | **Fortnite** | 0.795 |
| 2 | **Super Mario Odyssey** | 0.790 |
| 3 | **Cuphead** | 0.777 |
| 4 | **Ghost of Tsushima** | 0.773 |
| 5 | **Fall Guys** | 0.765 |

### 5. Sample Association Rules (`Apriori`, $\text{Confidence} = 1.00$)

* `{Street Fighter V, Call of Duty: Modern Warfare 2, Fall Guys, Portal 2}` $\rightarrow$ `{Kingdom Hearts III, Mario Kart 8 Deluxe}` ($\text{Lift} = 9.26$)
* `{Red Dead Redemption 2, Just Dance 2024, Grand Theft Auto V, The Elder Scrolls V: Skyrim}` $\rightarrow$ `{Hades}` ($\text{Lift} = 3.70$)
* `{The Legend of Zelda: Breath of the Wild, Tekken 7, Call of Duty: Modern Warfare 2, Half-Life: Alyx}` $\rightarrow$ `{Rocket League}` ($\text{Lift} = 3.65$)

---

## Visual Outputs

All generated charts are automatically saved in the `plots/` directory upon execution:

* **EDA & Feature Engineering:** `plots/eda_numeric_distributions.png`, `plots/eda_numeric_boxplots.png`, `plots/eda_correlation_matrix.png`, `plots/eda_top_tfidf_terms.png`
* **Regression & Classification:** `plots/reg_pca_vs_nopca_comparison.png`, `plots/reg_loss_without_pca.png`, `plots/reg_actual_vs_pred_without_pca.png`, `plots/cls_confusion_matrix.png`, `plots/cls_roc_curve.png`
* **Clustering:** `plots/clust_kmeans_elbow.png`, `plots/clust_final_convex_hull.png`, `plots/clust_feature_profiles.png`
* **Association Rules & Recommender:** `plots/assoc_top20_frequent_itemsets.png`, `plots/rec_top10_similar_games.png`

---

## Project Structure

```text
├── preprocessing.py                  # Data loading, EDA charts & multi-modal feature engineering
├── regression_and_classification.py  # Custom GD/SGD/Mini-Batch, Ridge, RF, GB, PCA & Logistic Regression
├── clustering.py                     # K-Means, Elbow/Silhouette analysis & 2D Convex Hull visualizations
├── association_and_recommender.py    # Apriori association rules & Cosine Similarity game recommender
├── 7000videogame.csv                 # Video game metadata & user reviews dataset (7,001 records)
├── basket_dataset.csv                # User-game transaction matrix for Apriori rule mining
├── plots/                            # Automatically exported high-resolution result figures
└── README.md
