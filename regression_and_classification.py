import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    classification_report, precision_recall_curve, average_precision_score,
    mean_squared_error, r2_score
)
from preprocessing import load_and_preprocess, target_col

os.makedirs("plots", exist_ok=True)

# بارگذاری داده پیش‌پردازش شده
df, final_df = load_and_preprocess(show_eda=False)


# ==================================================================
# ------------------- Linear Regression ----------------------------

def compute_mse(X, y, theta):
    return (1 / (2 * len(y))) * np.sum((X @ theta - y) ** 2)


def gradient_descent(X, y, theta, lr=0.01, epochs=200):
    m = len(y)
    losses = []
    for epoch in range(epochs):
        gradients = (1 / m) * X.T @ (X @ theta - y)
        theta -= lr * gradients
        losses.append(compute_mse(X, y, theta))
    return theta, losses


def stochastic_gradient_descent(X, y, theta, lr=0.017, epochs=200):
    m = len(y)
    losses = []
    for epoch in range(epochs):
        for i in range(m):
            idx = np.random.randint(m)
            xi = X[idx:idx + 1]
            yi = y[idx:idx + 1]
            gradients = xi.T @ (xi @ theta - yi)
            theta -= lr * gradients
        losses.append(compute_mse(X, y, theta))
    return theta, losses


def mini_batch_gradient_descent(X, y, theta, lr=0.029, epochs=200, batch_size=32):
    m = len(y)
    losses = []
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]
            gradients = (1 / len(yi)) * xi.T @ (xi @ theta - yi)
            theta -= lr * gradients
        losses.append(compute_mse(X, y, theta))
    return theta, losses


# --- آماده‌سازی داده ---
X = final_df.drop(columns=[target_col]).values
y = final_df[target_col].values.reshape(-1, 1)

scaler_reg = StandardScaler()
X_scaled = scaler_reg.fit_transform(X)

# --- Split train/test ---
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# اضافه کردن بایاس
X_train = np.c_[np.ones(X_train_raw.shape[0]), X_train_raw]
X_test = np.c_[np.ones(X_test_raw.shape[0]), X_test_raw]


# --- تعریف توابع برای اجرای سه روش ---
def run_all_methods(X_tr_b, X_te_b, y_tr, y_te):
    theta_init = np.zeros((X_tr_b.shape[1], 1))

    theta_batch, _ = gradient_descent(X_tr_b, y_tr, theta_init.copy())
    y_pred_batch = X_te_b @ theta_batch

    theta_sgd, _ = stochastic_gradient_descent(X_tr_b, y_tr, theta_init.copy())
    y_pred_sgd = X_te_b @ theta_sgd

    theta_mbgd, _ = mini_batch_gradient_descent(X_tr_b, y_tr, theta_init.copy())
    y_pred_mbgd = X_te_b @ theta_mbgd

    results = {}
    for name, y_pred in zip(["BatchGD", "SGD", "MiniBatch"], [y_pred_batch, y_pred_sgd, y_pred_mbgd]):
        mse_val = np.mean((y_te - y_pred) ** 2)
        r2_val = 1 - np.sum((y_te - y_pred) ** 2) / np.sum((y_te - np.mean(y_te)) ** 2)
        results[name] = {"mse": mse_val, "r2": r2_val}
    return results


# --- حالت بدون PCA ---
results_no_pca = run_all_methods(X_train, X_test, y_train, y_test)

# --- حالت با PCA ---
pca = PCA(n_components=0.95, random_state=42)  # حفظ 95% واریانس
X_pca_scaled = pca.fit_transform(X_scaled)
X_train_pca_raw, X_test_pca_raw, y_train_pca, y_test_pca = train_test_split(X_pca_scaled, y, test_size=0.3, random_state=42)
X_train_pca = np.c_[np.ones(X_train_pca_raw.shape[0]), X_train_pca_raw]
X_test_pca = np.c_[np.ones(X_test_pca_raw.shape[0]), X_test_pca_raw]
results_pca = run_all_methods(X_train_pca, X_test_pca, y_train_pca, y_test_pca)

# --- چاپ نتایج ---
print("Results without PCA:")
for k, v in results_no_pca.items():
    print(f"{k}: MSE={v['mse']:.4f}, R2={v['r2']:.4f}")

print("\nResults with PCA:")
for k, v in results_pca.items():
    print(f"{k}: MSE={v['mse']:.4f}, R2={v['r2']:.4f}")

# --- بررسی کاهش دقت با PCA ---
for method in ["BatchGD", "SGD", "MiniBatch"]:
    mse_no = results_no_pca[method]["mse"]
    mse_pca = results_pca[method]["mse"]
    r2_no = results_no_pca[method]["r2"]
    r2_pca = results_pca[method]["r2"]

    if mse_pca > mse_no:
        print(f"{method}: PCA decreased accuracy → MSE: {mse_no:.4f} → {mse_pca:.4f}, R²: {r2_no:.4f} → {r2_pca:.4f}")
    else:
        print(f"{method}: PCA did NOT decrease accuracy → MSE: {mse_no:.4f} → {mse_pca:.4f}, R²: {r2_no:.4f} → {r2_pca:.4f}")

# --- نمودار مقایسه‌ای ---
methods = ["BatchGD", "SGD", "MiniBatch"]
mse_no = [results_no_pca[m]["mse"] for m in methods]
mse_pca = [results_pca[m]["mse"] for m in methods]
r2_no = [results_no_pca[m]["r2"] for m in methods]
r2_pca = [results_pca[m]["r2"] for m in methods]

x = np.arange(len(methods))
width = 0.35

plt.figure(figsize=(12, 4))

# MSE
plt.subplot(1, 2, 1)
plt.bar(x - width / 2, mse_no, width, label="Without PCA", color="skyblue")
plt.bar(x + width / 2, mse_pca, width, label="With PCA", color="orange")
plt.xticks(x, methods)
plt.ylabel("MSE")
plt.title("Test MSE Comparison")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# R²
plt.subplot(1, 2, 2)
plt.bar(x - width / 2, r2_no, width, label="Without PCA", color="skyblue")
plt.bar(x + width / 2, r2_pca, width, label="With PCA", color="orange")
plt.xticks(x, methods)
plt.ylabel("R²")
plt.title("Test R² Comparison")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("plots/reg_pca_vs_nopca_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

theta_init = np.zeros((X_train.shape[1], 1))
theta_batch, cost_batch = gradient_descent(X_train, y_train, theta_init.copy())
theta_sgd, cost_sgd = stochastic_gradient_descent(X_train, y_train, theta_init.copy())
theta_mbgd, cost_mbgd = mini_batch_gradient_descent(X_train, y_train, theta_init.copy())

theta_init_pca = np.zeros((X_train_pca.shape[1], 1))
theta_batch_pca, cost_batch_pca = gradient_descent(X_train_pca, y_train_pca, theta_init_pca.copy())
theta_sgd_pca, cost_sgd_pca = stochastic_gradient_descent(X_train_pca, y_train_pca, theta_init_pca.copy())
theta_mbgd_pca, cost_mbgd_pca = mini_batch_gradient_descent(X_train_pca, y_train_pca, theta_init_pca.copy())

# رسم منحنی‌های loss
plt.figure(figsize=(8, 5))
plt.plot(cost_batch, label="Batch GD")
plt.plot(cost_sgd, label="Stochastic GD")
plt.plot(cost_mbgd, label="Mini-batch GD")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Linear Regression Loss without pca")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/reg_loss_without_pca.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(cost_batch_pca, label="Batch GD")
plt.plot(cost_sgd_pca, label="Stochastic GD")
plt.plot(cost_mbgd_pca, label="Mini-batch GD")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Linear Regression Loss with pca")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/reg_loss_with_pca.png", dpi=300, bbox_inches="tight")
plt.show()


# --- محاسبه معیارها (MSE و R2) برای هر مدل روی مجموعه تست ---
def mse_np(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan


models = {
    "BatchGD": theta_batch,
    "SGD": theta_sgd,
    "MiniBatch": theta_mbgd
}

results = {}
for name, theta in models.items():
    y_pred_test = X_test @ theta
    y_pred_train = X_train @ theta
    results[name] = {
        "mse_test": mse_np(y_test, y_pred_test),
        "mse_train": mse_np(y_train, y_pred_train),
        "r2_test": r2_np(y_test, y_pred_test),
        "r2_train": r2_np(y_train, y_pred_train),
        "y_pred_test": y_pred_test
    }

models_pca = {
    "BatchGD": theta_batch_pca,
    "SGD": theta_sgd_pca,
    "MiniBatch": theta_mbgd_pca
}

results_pca = {}
for name, theta_pca in models_pca.items():
    y_pred_test_pca = X_test_pca @ theta_pca
    y_pred_train_pca = X_train_pca @ theta_pca
    results_pca[name] = {
        "mse_test_pca": mse_np(y_test_pca, y_pred_test_pca),
        "mse_train_pca": mse_np(y_train_pca, y_pred_train_pca),
        "r2_test_pca": r2_np(y_test_pca, y_pred_test_pca),
        "r2_train_pca": r2_np(y_train_pca, y_pred_train_pca),
        "y_pred_test_pca": y_pred_test_pca
    }

# Ridge (برای مقایسه)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train[:, 1:], y_train)
y_ridge_test = ridge.predict(X_test[:, 1:])
results["Ridge"] = {
    "mse_test": mse_np(y_test, y_ridge_test),
    "r2_test": ridge.score(X_test[:, 1:], y_test),
    "y_pred_test": y_ridge_test
}

ridge_pca = Ridge(alpha=1.0)
ridge_pca.fit(X_train_pca[:, 1:], y_train_pca)
y_ridge_test_pca = ridge_pca.predict(X_test_pca[:, 1:])
results_pca["Ridge"] = {
    "mse_test_pca": mse_np(y_test_pca, y_ridge_test_pca),
    "r2_test_pca": ridge_pca.score(X_test_pca[:, 1:], y_test_pca),
    "y_pred_test_pca": y_ridge_test_pca
}

# چاپ جدول خلاصه معیارها
print("\nRegression results summary (test):")
for name, vals in results.items():
    print(f"{name}: MSE_test={vals['mse_test']:.6f}, R2_test={vals.get('r2_test', np.nan):.6f}")

# نمودار مقایسه MSE و R2
names = list(results.keys())
mse_vals = [results[n]["mse_test"] for n in names]
r2_vals = [results[n].get("r2_test", np.nan) for n in names]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(len(names)), mse_vals)
plt.xticks(range(len(names)), names, rotation=45)
plt.title("Test MSE by model")
plt.subplot(1, 2, 2)
plt.bar(range(len(names)), r2_vals)
plt.xticks(range(len(names)), names, rotation=45)
plt.title("Test R2 by model")
plt.tight_layout()
plt.savefig("plots/reg_models_metrics_without_pca.png", dpi=300, bbox_inches="tight")
plt.show()

# نمودار مقایسه MSE و R2 -> pca
names_pca = list(results_pca.keys())
mse_vals_pca = [results_pca[n]["mse_test_pca"] for n in names_pca]
r2_vals_pca = [results_pca[n].get("r2_test_pca", np.nan) for n in names_pca]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(len(names_pca)), mse_vals_pca)
plt.xticks(range(len(names_pca)), names_pca, rotation=45)
plt.title("Test MSE by model with pca")
plt.subplot(1, 2, 2)
plt.bar(range(len(names_pca)), r2_vals_pca)
plt.xticks(range(len(names_pca)), names_pca, rotation=45)
plt.title("Test R2 by model with pca")
plt.tight_layout()
plt.savefig("plots/reg_models_metrics_with_pca.png", dpi=300, bbox_inches="tight")
plt.show()

# نمودارهای پیش‌بینی vs واقعی روی مجموعه تست (سه مدل)
plt.figure(figsize=(15, 4))
for i, name in enumerate(["BatchGD", "SGD", "MiniBatch"]):
    plt.subplot(1, 3, i + 1)
    y_pred = results[name]["y_pred_test"].ravel()
    plt.scatter(y_test.ravel(), y_pred, s=8, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--')
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"{name}: Actual vs Pred (test)")
plt.tight_layout()
plt.savefig("plots/reg_actual_vs_pred_without_pca.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(15, 4))
for i, name in enumerate(["BatchGD", "SGD", "MiniBatch"]):
    plt.subplot(1, 3, i + 1)
    y_pred_pca = results_pca[name]["y_pred_test_pca"].ravel()
    plt.scatter(y_test_pca.ravel(), y_pred_pca, s=8, alpha=0.4)
    plt.plot([y_test_pca.min(), y_test_pca.max()], [y_test_pca.min(), y_test_pca.max()], '--')
    plt.xlabel("Actual with pca")
    plt.ylabel("Predicted Rating with pca")
    plt.title(f"{name}: Actual vs Pred (test) with pca")
plt.tight_layout()
plt.savefig("plots/reg_actual_vs_pred_with_pca.png", dpi=300, bbox_inches="tight")
plt.show()

# باقیمانده‌ها (Residuals) هیستوگرام برای هر مدل
plt.figure(figsize=(12, 4))
for i, name in enumerate(["BatchGD", "SGD", "MiniBatch"]):
    plt.subplot(1, 3, i + 1)
    residuals = (y_test - results[name]["y_pred_test"]).ravel()
    plt.hist(residuals, bins=40)
    plt.title(f"{name} residuals (test)")
plt.tight_layout()
plt.savefig("plots/reg_residuals_histograms.png", dpi=300, bbox_inches="tight")
plt.show()

# نمودار ضرایب (weights) برای مدل Batch (مهم‌ترین‌ها)
feature_names = final_df.drop(columns=[target_col]).columns.tolist()
theta_vals = theta_batch.ravel()
coef = theta_vals[1:]
feat = feature_names
abs_idx = np.argsort(np.abs(coef))[-20:][::-1]
plt.figure(figsize=(10, 4))
plt.bar(range(len(abs_idx)), coef[abs_idx])
plt.xticks(range(len(abs_idx)), [feat[i] for i in abs_idx], rotation=90)
plt.title("Top 20 coefficients (Batch GD)")
plt.tight_layout()
plt.savefig("plots/reg_top20_coefficients.png", dpi=300, bbox_inches="tight")
plt.show()

# ==================================================================
# ------------ Ensemble Regressors (RandomForest & GB) -------------
tree_models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
}

tree_results = {}
for name, model in tree_models.items():
    model.fit(X_train_raw, y_train.ravel())
    y_pred_tr = model.predict(X_train_raw)
    y_pred_te = model.predict(X_test_raw)
    tree_results[name] = {
        "mse_train": mean_squared_error(y_train.ravel(), y_pred_tr),
        "mse_test": mean_squared_error(y_test.ravel(), y_pred_te),
        "r2_train": r2_score(y_train.ravel(), y_pred_tr),
        "r2_test": r2_score(y_test.ravel(), y_pred_te),
    }

print("\nEnsemble Regression results (Train/Test):")
for name, res in tree_results.items():
    print(f"{name}: MSE_train={res['mse_train']:.4f}, MSE_test={res['mse_test']:.4f}, "
          f"R2_train={res['r2_train']:.4f}, R2_test={res['r2_test']:.4f}")


# ==================================================================
# ------------------- Classification (Logistic Regression) ---------

threshold = final_df["User Rating"].mean()
y_class = (final_df["User Rating"] > threshold).astype(int).values.reshape(-1, 1)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled, y_class, test_size=0.3, random_state=42)
X_train_cls = np.c_[np.ones(X_train_cls.shape[0]), X_train_cls]
X_test_cls = np.c_[np.ones(X_test_cls.shape[0]), X_test_cls]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_log_loss(X, y, theta):
    y_pred = sigmoid(X @ theta)
    epsilon = 1e-8
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))


def logistic_regression(X, y, lr=0.01, epochs=200):
    m, n = X.shape
    theta = np.zeros((n, 1))
    losses = []
    for epoch in range(epochs):
        y_pred = sigmoid(X @ theta)
        gradients = (1 / m) * X.T @ (y_pred - y)
        theta -= lr * gradients
        losses.append(compute_log_loss(X, y, theta))
    return theta, losses


theta_log, losses_log = logistic_regression(X_train_cls, y_train_cls, lr=0.01, epochs=200)

# رسم loss
plt.figure(figsize=(6, 4))
plt.plot(losses_log)
plt.title("Logistic regression training loss")
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/cls_logistic_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# پیش‌بینی و ارزیابی
y_proba = sigmoid(X_test_cls @ theta_log)
y_pred_test = (y_proba >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test_cls, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Disliked", "Liked"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Logistic)")
plt.tight_layout()
plt.savefig("plots/cls_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# Classification report
print("\nClassification report (Logistic):")
print(classification_report(y_test_cls, y_pred_test))

# ROC curve + AUC
fpr, tpr, _ = roc_curve(y_test_cls, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0, 1], [0, 1], '--')
plt.title("ROC Curve (Logistic)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/cls_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test_cls, y_proba)
avg_prec = average_precision_score(y_test_cls, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(recall, precision, label=f"AP={avg_prec:.3f}")
plt.title("Precision-Recall (Logistic)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/cls_precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.show()