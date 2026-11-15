import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

os.makedirs("plots", exist_ok=True)

# ---------------- تعریف ستون‌ها ----------------
DATA_PATH = "7000videogame.csv"
target_col = "User Rating"
low_cardinality_cols = [
    "Age Group Targeted", "Platform", "Requires Special Device", "Genre",
    "Multiplayer", "Graphics Quality", "Soundtrack Quality", "Story Quality", "Game Mode"
]
numeric_cols = ["Release Year", "Game Length Hours", "Min Number of Players"]
high_cardinality_cols = ["Publisher", "Developer"]
text_col = "User Review Text"


def load_and_preprocess(file_path=DATA_PATH, show_eda=True):
    # ---------------- بارگذاری داده ----------------
    df = pd.read_csv(file_path)

    if show_eda:
        # ---------------- ================ EDA اولیه =======================
        print("Dataset shape:", df.shape)
        print("Columns:", df.columns.tolist())

        # توزیع ستون‌های عددی (قبل از هر گونه تغییر)
        plt.figure(figsize=(12, 4))
        for i, col in enumerate(numeric_cols):
            plt.subplot(1, len(numeric_cols), i + 1)
            if col in df.columns:
                plt.hist(df[col].dropna(), bins=30)
                plt.title(f"Distribution: {col}")
                plt.xlabel(col)
        plt.tight_layout()
        plt.savefig("plots/eda_numeric_distributions.png", dpi=300, bbox_inches="tight")
        plt.show()

        # آمار خلاصه اعداد
        print("\nNumeric summary:")
        print(df[numeric_cols].describe())

        # مقادیر پرت و boxplot
        plt.figure(figsize=(12, 4))
        for i, col in enumerate(numeric_cols):
            plt.subplot(1, len(numeric_cols), i + 1)
            if col in df.columns:
                plt.boxplot(df[col].dropna(), vert=True)
                plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig("plots/eda_numeric_boxplots.png", dpi=300, bbox_inches="tight")
        plt.show()

        # ستون‌های دسته‌ای: نمودار top categories برای هر low_cardinality_col
        for col in low_cardinality_cols:
            if col in df.columns:
                vc = df[col].value_counts().nlargest(10)
                plt.figure(figsize=(7, 3))
                plt.bar(range(len(vc)), vc.values)
                plt.xticks(range(len(vc)), vc.index, rotation=45, ha="right")
                plt.title(f"Top categories in {col}")
                plt.tight_layout()
                plt.savefig(f"plots/eda_top_categories_{col.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
                plt.show()

    # ---------------- One-Hot Encoding ----------------
    try:
        ohe = OneHotEncoder(sparse_output=False, drop="first")
    except TypeError:
        ohe = OneHotEncoder(sparse=False, drop="first")

    encoded_array = ohe.fit_transform(df[low_cardinality_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(low_cardinality_cols), index=df.index)

    # ---------------- Scaling برای ویژگی‌های عددی ----------------
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_cols, index=df.index)

    if show_eda:
        # نمایش قبل/بعد استانداردسازی (برای یک ستون نمونه)
        for col in numeric_cols:
            if col in df.columns:
                plt.figure(figsize=(8, 3))
                plt.subplot(1, 2, 1)
                plt.hist(df[col].dropna(), bins=30)
                plt.title(f"Raw: {col}")
                plt.subplot(1, 2, 2)
                plt.hist(scaled_df[col].dropna(), bins=30)
                plt.title(f"Scaled: {col}")
                plt.tight_layout()
                plt.savefig(f"plots/eda_scaling_{col.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
                plt.show()

        # همبستگی بین ستون‌های عددی (روی داده اصلی)
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            plt.figure(figsize=(5, 4))
            plt.imshow(corr, interpolation='nearest')
            plt.colorbar()
            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
            plt.yticks(range(len(numeric_cols)), numeric_cols)
            plt.title("Correlation matrix (numeric cols)")
            for (i, j), val in np.ndenumerate(corr.values):
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8)
            plt.tight_layout()
            plt.savefig("plots/eda_correlation_matrix.png", dpi=300, bbox_inches="tight")
            plt.show()

    # ---------------- Frequency Encoding برای Publisher و Developer ----------------
    freq_df = pd.DataFrame()
    for col in high_cardinality_cols:
        if col in df.columns:
            freq_map = df[col].value_counts().to_dict()
            freq_df[col + "_freq"] = df[col].map(freq_map)

    if show_eda:
        # نشان دادن توزیع فراوانی publisher/developer (top 20)
        for col in high_cardinality_cols:
            if col in df.columns:
                vc = df[col].value_counts().nlargest(20)
                plt.figure(figsize=(8, 3))
                plt.bar(range(len(vc)), vc.values)
                plt.xticks(range(len(vc)), vc.index, rotation=45, ha='right')
                plt.title(f"Top 20 counts for {col}")
                plt.tight_layout()
                plt.savefig(f"plots/eda_top20_{col.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
                plt.show()

    # ---------------- TF-IDF برای متن ----------------
    tfidf = TfidfVectorizer(max_features=100)
    if text_col in df.columns:
        tfidf_array = tfidf.fit_transform(df[text_col].fillna(""))
        tfidf_df = pd.DataFrame(
            tfidf_array.toarray(),
            columns=[f"tfidf_{i}" for i in range(tfidf_array.shape[1])],
            index=df.index
        )

        if show_eda:
            # ترم‌های برتر بر اساس میانگین tfidf
            feature_names = tfidf.get_feature_names_out()
            mean_tfidf = np.asarray(tfidf_array.mean(axis=0)).ravel()
            top_idx = np.argsort(mean_tfidf)[-20:][::-1]
            top_terms = feature_names[top_idx]
            top_vals = mean_tfidf[top_idx]

            plt.figure(figsize=(10, 4))
            plt.bar(range(len(top_terms)), top_vals)
            plt.xticks(range(len(top_terms)), top_terms, rotation=45, ha='right')
            plt.title("Top TF-IDF terms (mean across docs)")
            plt.tight_layout()
            plt.savefig("plots/eda_top_tfidf_terms.png", dpi=300, bbox_inches="tight")
            plt.show()
    else:
        tfidf_df = pd.DataFrame(index=df.index)

    # ---------------- ترکیب همه داده‌ها ----------------
    to_concat = [scaled_df, encoded_df, freq_df, tfidf_df]
    to_concat = [t for t in to_concat if t.shape[0] == df.shape[0]]
    final_df = pd.concat(to_concat + [df[[target_col]]], axis=1)

    print("Final shape:", final_df.shape)
    return df, final_df


if __name__ == "__main__":
    load_and_preprocess(show_eda=True)