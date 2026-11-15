import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "7000videogame.csv")
BASKET_PATH = os.path.join(BASE_DIR, "basket_dataset.csv")

# ==================================================================
# ------------------- Association Rules ----------------------------

df = pd.read_csv(DATA_PATH)

if os.path.exists(BASKET_PATH):
    basket_loaded = pd.read_csv(BASKET_PATH, index_col=0)
else:
    # اضافه کردن ستون User ID تصادفی و ساخت ماتریس سبد خرید
    df_basket = df.copy()
    np.random.seed(42)
    num_users = 500
    df_basket["User ID"] = np.random.randint(1, num_users + 1, size=len(df_basket))
    basket_loaded = df_basket.groupby(["User ID", "Game Title"]).size().unstack().fillna(0)
    basket_loaded = (basket_loaded > 0).astype(int)
    basket_loaded.to_csv(BASKET_PATH, index=True)
    print(f"Basket dataset saved as {BASKET_PATH}")

# اجرای Apriori
min_support_val = 0.01
frequent_itemsets = apriori(basket_loaded.astype(bool), min_support=min_support_val, use_colnames=True)

if not frequent_itemsets.empty:
    # نمایش top 20 frequent itemsets
    topn = frequent_itemsets.sort_values("support", ascending=False).head(20)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(topn)), topn["support"])
    plt.xticks(range(len(topn)), [','.join(list(x)) for x in topn['itemsets']], rotation=90)
    plt.title(f"Top 20 frequent itemsets (support ≥ {min_support_val})")
    plt.tight_layout()
    out1 = os.path.join(PLOTS_DIR, "assoc_top20_frequent_itemsets.png")
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.show()

    # استخراج قوانین
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    if not rules.empty:
        top_rules = rules.sort_values("confidence", ascending=False).head(10)
        print("\nTop 10 Association Rules by confidence:")
        print(top_rules[["antecedents", "consequents", "support", "confidence", "lift"]])

        # تفسیر قوانین
        print("\n=== Rule Interpretations ===")
        for _, row in top_rules.iterrows():
            antecedent = ", ".join(list(row["antecedents"]))
            consequent = ", ".join(list(row["consequents"]))
            print(f"If a user has selected {antecedent} → there is a {row['confidence']:.2f} probability they will also select {consequent} (lift={row['lift']:.2f})")
else:
    print("No frequent itemsets found with the given min_support.")


# ==================================================================
# ------------- Recommender System (Content-based) -----------------

low_cardinality_cols = [
    "Age Group Targeted", "Platform", "Requires Special Device", "Genre",
    "Multiplayer", "Graphics Quality", "Soundtrack Quality", "Story Quality", "Game Mode"
]
numeric_cols = ["Release Year", "Game Length Hours", "Min Number of Players"]
text_col = "User Review Text"

scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)

try:
    ohe = OneHotEncoder(sparse_output=False, drop="first")
except TypeError:
    ohe = OneHotEncoder(sparse=False, drop="first")

encoded_array = ohe.fit_transform(df[low_cardinality_cols])
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(low_cardinality_cols), index=df.index)

if text_col in df.columns:
    tfidf = TfidfVectorizer(max_features=100)
    tfidf_array = tfidf.fit_transform(df[text_col].fillna(""))
    tfidf_df = pd.DataFrame(tfidf_array.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_array.shape[1])], index=df.index)
else:
    tfidf_df = pd.DataFrame(index=df.index)

feature_list = [scaled_df.values, encoded_df.values]
if not tfidf_df.empty:
    feature_list.append(tfidf_df.values)

X_features = np.hstack(feature_list)

# نرمال‌سازی ویژگی‌ها
row_norms = np.linalg.norm(X_features, axis=1)
nonzero_idx = row_norms > 0
X_norm = np.zeros_like(X_features)
X_norm[nonzero_idx] = X_features[nonzero_idx] / row_norms[nonzero_idx][:, None]

# شباهت کسینوسی
similarity_matrix = cosine_similarity(X_norm)

if "Game Title" in df.columns:
    game_titles = df["Game Title"].values
else:
    game_titles = np.array([f"Game_{i}" for i in range(len(df))])

title_to_idx = {title: i for i, title in enumerate(game_titles)}


def recommend_similar_games(game_name, top_n=10):
    if game_name not in title_to_idx:
        raise ValueError(f"{game_name} not found in dataset.")
    idx = title_to_idx[game_name]
    sim_scores = similarity_matrix[idx].copy()
    sim_scores[idx] = -1  # حذف خود بازی از پیشنهادها
    top_idx = np.argsort(sim_scores)[-top_n:][::-1]
    recommendations = [(game_titles[i], sim_scores[i]) for i in top_idx]
    return recommendations


sample_game = "Grand Theft Auto V"
recs = recommend_similar_games(sample_game, top_n=10)

print(f"\nTop 10 games similar to '{sample_game}':")
for title, score in recs:
    print(f"{title}: similarity={score:.3f}")

rec_titles = [t for t, s in recs]
rec_scores = [s for t, s in recs]

plt.figure(figsize=(8, 4))
plt.bar(range(len(rec_titles)), rec_scores, color="skyblue")
plt.xticks(range(len(rec_titles)), rec_titles, rotation=45, ha='right')
plt.ylabel("Cosine similarity")
plt.title(f"Top {len(rec_titles)} similar games to '{sample_game}'")
plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, "rec_top10_similar_games.png")
plt.savefig(out2, dpi=300, bbox_inches="tight")
print(f"Saved: {out2}")
plt.show()