# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (7, 6)
})

print("\nImports: scikit-learn version")
print("scikit-learn version:", sklearn.__version__)

# Folder where all figures/plots are saved
os.makedirs("figures", exist_ok=True)

# Dataset Identifier
data_path = "week2.csv"

with open(data_path, "r") as file:
    dataset_identifier = file.readline().strip()

print("\nDataset Identifier (first line of the CSV)")
print("Dataset identifier:", dataset_identifier)

# Load the Dataset
df = pd.read_csv(
    data_path,
    comment="#",
    header=None,
    names=["x1", "x2", "target"]
)

print("\nLoad the Dataset: first 5 rows, shape and class counts")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nClass counts:")
print(df["target"].value_counts())

# Prepare input features X and Y
X = df[["x1", "x2"]].to_numpy()
y = df["target"].to_numpy()

print("\nPrepare input features X and y: shapes and first 5 values")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 feature pairs:")
print(X[:5])

print("\nFirst 5 targets:")
print(y[:5])

# Part (a)(i): Visualize original data
print("\nPart (a)(i): Visualize original data (plot saved: figures/a_i_training_data.png)")
positive = y == 1
negative = y == -1

plt.figure(figsize=(8, 6))

# +1 points
plt.scatter(
    X[positive, 0],
    X[positive, 1],
    marker="+",
    s=50,
    label="+1"
)

# -1 points
plt.scatter(
    X[negative, 0],
    X[negative, 1],
    marker="o",
    s=50,
    label="-1"
)

plt.xlabel("Feature 1 (x1)")
plt.ylabel("Feature 2 (x2)")
plt.title("Training Data")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("figures/a_i_training_data.png", dpi=150, bbox_inches="tight")

# Part (a)(ii): Train logistic regression
print("\nPart (a)(ii): Train logistic regression - intercept, weights and decision function")
logistic_model = LogisticRegression(max_iter=10000)

logistic_model.fit(X, y)

log_intercept = logistic_model.intercept_[0]
log_weights = logistic_model.coef_[0]

print("Logistic Regression Parameters")
print("--------------------------------")
print("Intercept =", log_intercept)
print("Weight for x1 =", log_weights[0])
print("Weight for x2 =", log_weights[1])

print(
    f"\nDecision function:\n"
    f"z = {log_intercept:.6f} "
    f"+ ({log_weights[0]:.6f})x1 "
    f"+ ({log_weights[1]:.6f})x2"
)

# Logistic regression accuracy
print("\nPart (a)(ii) continued: Logistic regression accuracy and confusion matrix")
log_predictions = logistic_model.predict(X)

log_accuracy = accuracy_score(y, log_predictions)

print("Logistic Regression Accuracy:")
print(f"{log_accuracy:.4f}")
print(f"{log_accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y, log_predictions))

# Part (a)(iii): Plot of Logistic Regression showing actual Vs Predicted
print("\nPart (a)(iii): Logistic regression actual vs predicted (plot saved: figures/a_iii_logistic_actual_vs_predicted.png)")
w1, w2 = logistic_model.coef_[0]
b = logistic_model.intercept_[0]

x1_line = np.linspace(
    X[:, 0].min() - 0.1,
    X[:, 0].max() + 0.1,
    300
)

x2_line = -(b + w1 * x1_line) / w2

actual_positive = y == 1
actual_negative = y == -1

predicted_positive = log_predictions == 1
predicted_negative = log_predictions == -1

plt.figure(figsize=(8, 7))

# Actual observations
plt.scatter(
    X[actual_positive, 0],
    X[actual_positive, 1],
    marker="+",
    s=60,
    label="Actual +1"
)

plt.scatter(
    X[actual_negative, 0],
    X[actual_negative, 1],
    marker="o",
    s=50,
    facecolors="none",
    edgecolors="red",
    linewidths=1.5,
    label="Actual -1"
)

# Predictions
plt.scatter(
    X[predicted_positive, 0],
    X[predicted_positive, 1],
    marker="^",
    s=15,
    label="Predicted +1"
)

plt.scatter(
    X[predicted_negative, 0],
    X[predicted_negative, 1],
    marker="x",
    s=25,
    label="Predicted -1"
)

# Decision boundary
plt.plot(
    x1_line,
    x2_line,
    linewidth=2,
    label="Decision Boundary"
)

plt.xlabel("Feature 1 (x1)")
plt.ylabel("Feature 2 (x2)")
plt.title("Logistic Regression: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("figures/a_iii_logistic_actual_vs_predicted.png", dpi=150, bbox_inches="tight")

# Part (a)(iv): Number of correct/incorrect predictions
print("\nPart (a)(iv): Number of correct / incorrect predictions")
correct_predictions = np.sum(log_predictions == y)
incorrect_predictions = np.sum(log_predictions != y)

print("Total observations:", len(y))
print("Correct predictions:", correct_predictions)
print("Incorrect predictions:", incorrect_predictions)
print(f"Accuracy: {log_accuracy * 100:.2f}%")

# Part (b)(i): Train SVM models
print("\nPart (b)(i): Train linear SVM models for C = 0.001, 1, 100")
C_values = [0.001, 1, 100]

svm_models = {}

for C in C_values:
    
    model = LinearSVC(
        C=C,
        max_iter=100000
    )
    
    model.fit(X, y)
    
    svm_models[C] = model
    
    b = model.intercept_[0]
    w1, w2 = model.coef_[0]
    
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    print(f"C = {C}")
    print(f"Intercept = {b:.6f}")
    print(f"Weight x1 = {w1:.6f}")
    print(f"Weight x2 = {w2:.6f}")
    print(f"Accuracy = {accuracy:.4f}")
    print(
        f"Decision function: "
        f"{b:.6f} + ({w1:.6f})x1 + ({w2:.6f})x2"
    )
    print("-" * 50)

# Part (b)(ii): SVM C = 0.001, 1, 100
print("\nPart (b)(ii): SVM plots for C = 0.001, 1, 100 (plot saved: figures/b_ii_svm_C_comparison.png)")
C_values = [0.001, 1, 100]

x1_line = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 300)

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

for ax, C in zip(axes, C_values):
    model = svm_models[C]
    svm_predictions = model.predict(X)

    w1, w2 = model.coef_[0]
    b = model.intercept_[0]
    x2_line = -(b + w1 * x1_line) / w2

    actual_positive = y == 1
    actual_negative = y == -1
    predicted_positive = svm_predictions == 1
    predicted_negative = svm_predictions == -1

    ax.scatter(X[actual_positive, 0], X[actual_positive, 1],
               marker="+", s=60, label="Actual +1")
    ax.scatter(X[actual_negative, 0], X[actual_negative, 1],
               marker="o", s=40, facecolors="none", edgecolors="red",
               linewidths=1.5, label="Actual -1")
    ax.scatter(X[predicted_positive, 0], X[predicted_positive, 1],
               marker=".", s=15, label="Predicted +1")
    ax.scatter(X[predicted_negative, 0], X[predicted_negative, 1],
               marker="x", s=25, label="Predicted -1")
    ax.plot(x1_line, x2_line, linewidth=2, color="black", label="Decision Boundary")

    ax.set_ylim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    ax.set_xlabel("Feature 1 (x1)")
    ax.set_title(f"Linear SVM, C = {C}")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Feature 2 (x2)")
axes[0].legend()
plt.tight_layout()
plt.savefig("figures/b_ii_svm_C_comparison.png", dpi=150, bbox_inches="tight")

# Part (b)(iii)/(iv): Comparison table
print("\nPart (b)(iii)/(iv): Comparison table of Logistic Regression and SVM models")
results = []

results.append({
    "Model": "Logistic Regression",
    "C": "-",
    "Intercept": logistic_model.intercept_[0],
    "Weight x1": logistic_model.coef_[0][0],
    "Weight x2": logistic_model.coef_[0][1],
    "Accuracy": accuracy_score(y, logistic_model.predict(X))
})

for C, model in svm_models.items():
    
    results.append({
        "Model": "Linear SVM",
        "C": C,
        "Intercept": model.intercept_[0],
        "Weight x1": model.coef_[0][0],
        "Weight x2": model.coef_[0][1],
        "Accuracy": accuracy_score(y, model.predict(X))
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# PART (c) — Adding squared features

# Part (c)(i): Create squared features
print("\nPart (c)(i): Create squared features - new shape and first 5 rows")
x1 = X[:, 0]
x2 = X[:, 1]

X_squared = np.column_stack((
    x1,
    x2,
    x1 ** 2,
    x2 ** 2
))

print("Original X shape:", X.shape)
print("New X shape:", X_squared.shape)

print("\nFirst 5 rows:")
print(X_squared[:5])

# Part(c)(ii): Training Logistic regression with 4 inputs
print("\nPart (c)(ii): Logistic regression with 4 inputs - intercept, weights and decision function")
quadratic_model = LogisticRegression(max_iter=10000)

quadratic_model.fit(X_squared, y)

b = quadratic_model.intercept_[0]
weights = quadratic_model.coef_[0]

print("Logistic Regression with Squared Features")
print("------------------------------------------")

print(f"Intercept = {b:.6f}")
print(f"Weight x1 = {weights[0]:.6f}")
print(f"Weight x2 = {weights[1]:.6f}")
print(f"Weight x1^2 = {weights[2]:.6f}")
print(f"Weight x2^2 = {weights[3]:.6f}")

print("\nDecision function:")

print(
    f"z = {b:.6f} "
    f"+ ({weights[0]:.6f})x1 "
    f"+ ({weights[1]:.6f})x2 "
    f"+ ({weights[2]:.6f})x1^2 "
    f"+ ({weights[3]:.6f})x2^2"
)

# Part (c)(ii): Predictions
print("\nPart (c)(ii) continued: Accuracy and confusion matrix with squared features")
quadratic_predictions = quadratic_model.predict(X_squared)

quadratic_accuracy = accuracy_score(
    y,
    quadratic_predictions
)

print("Accuracy with squared features:")
print(f"{quadratic_accuracy:.4f}")
print(f"{quadratic_accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y, quadratic_predictions))

# Plot actual and predicted classes
print("\nPart (c)(ii) plot: Actual vs predicted with squared features (plot saved: figures/c_ii_squared_features_actual_vs_predicted.png)")
actual_positive = y == 1
actual_negative = y == -1

predicted_positive = quadratic_predictions == 1
predicted_negative = quadratic_predictions == -1

plt.figure(figsize=(8, 7))

# Actual data
plt.scatter(
    X[actual_positive, 0],
    X[actual_positive, 1],
    marker="+",
    s=60,
    label="Actual +1"
)

plt.scatter(
    X[actual_negative, 0],
    X[actual_negative, 1],
    marker="o",
    s=40,
    facecolors="none",
    edgecolors="red",
    linewidths=1.5,
    label="Actual -1"
)

# Predictions
plt.scatter(
    X[predicted_positive, 0],
    X[predicted_positive, 1],
    marker=".",
    s=15,
    label="Predicted +1"
)

plt.scatter(
    X[predicted_negative, 0],
    X[predicted_negative, 1],
    marker="x",
    s=25,
    label="Predicted -1"
)

plt.xlabel("Feature 1 (x1)")
plt.ylabel("Feature 2 (x2)")
plt.title("Logistic Regression with Squared Features")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("figures/c_ii_squared_features_actual_vs_predicted.png", dpi=150, bbox_inches="tight")

# Compare original logistic regression vs squared features
print("\nCompare original logistic regression vs squared features")
original_accuracy = accuracy_score(
    y,
    logistic_model.predict(X)
)

quadratic_accuracy = accuracy_score(
    y,
    quadratic_model.predict(X_squared)
)

print("Original Logistic Regression:")
print(f"{original_accuracy * 100:.2f}%")

print("\nLogistic Regression with Squared Features:")
print(f"{quadratic_accuracy * 100:.2f}%")

print(
    "\nImprovement:",
    f"{(quadratic_accuracy - original_accuracy) * 100:.2f} percentage points"
)

# Part (c)(iii): Baseline predictor
print("\nPart (c)(iii): Baseline predictor (most frequent class)")
baseline_model = DummyClassifier(
    strategy="most_frequent"
)

baseline_model.fit(X_squared, y)

baseline_predictions = baseline_model.predict(X_squared)

baseline_accuracy = accuracy_score(
    y,
    baseline_predictions
)

print("Most common class:", baseline_model.classes_[
    np.argmax(baseline_model.class_prior_)
])

print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline Accuracy: {baseline_accuracy * 100:.2f}%")

print(
    f"Squared-feature Logistic Regression Accuracy: "
    f"{quadratic_accuracy * 100:.2f}%"
)

# Final model accuracy comparison
print("\nFinal model accuracy comparison table")
comparison = pd.DataFrame({
    "Model": [
        "Baseline (Most Frequent)",
        "Logistic Regression",
        "Linear SVM C=0.001",
        "Linear SVM C=1",
        "Linear SVM C=100",
        "Logistic Regression + Squared Features"
    ],
    "Accuracy": [
        baseline_accuracy,
        accuracy_score(y, logistic_model.predict(X)),
        accuracy_score(y, svm_models[0.001].predict(X)),
        accuracy_score(y, svm_models[1].predict(X)),
        accuracy_score(y, svm_models[100].predict(X)),
        quadratic_accuracy
    ]
})

comparison["Accuracy (%)"] = comparison["Accuracy"] * 100

print(comparison.to_string(index=False))

# Part (c)(iv) Non Linear Decision Boundary
print("\nPart (c)(iv): Non linear decision boundary (plot saved: figures/c_iv_nonlinear_decision_boundary.png)")
b = quadratic_model.intercept_[0]
w1, w2, w3, w4 = quadratic_model.coef_[0]

x1_line = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 500)

c = b + w1 * x1_line + w3 * x1_line ** 2
discriminant = w2 ** 2 - 4 * w4 * c

valid = discriminant >= 0
sqrt_disc = np.sqrt(discriminant[valid])

x1_valid = x1_line[valid]
x2_root_plus = (-w2 + sqrt_disc) / (2 * w4)
x2_root_minus = (-w2 - sqrt_disc) / (2 * w4)

x2_low = X[:, 1].min() - 0.1
x2_high = X[:, 1].max() + 0.1

plt.figure(figsize=(8, 7))

plt.scatter(X[y == 1, 0], X[y == 1, 1],
            marker="+", s=60, label="Actual +1")

plt.scatter(X[y == -1, 0], X[y == -1, 1],
            marker="o", s=40, facecolors="none",
            edgecolors="red", linewidths=1.5, label="Actual -1")

labelled = False
for root in (x2_root_plus, x2_root_minus):
    inside = (root >= x2_low) & (root <= x2_high)
    if inside.any():
        plt.plot(x1_valid[inside], root[inside],
                 color="black", linewidth=2,
                 label=None if labelled else "Decision Boundary")
        labelled = True

plt.xlim(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
plt.ylim(x2_low, x2_high)

plt.xlabel("Feature 1 (x1)")
plt.ylabel("Feature 2 (x2)")
plt.title("Nonlinear Decision Boundary (Quadratic Formula)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/c_iv_nonlinear_decision_boundary.png", dpi=150, bbox_inches="tight")

# Show all figures together at the end
plt.show()
