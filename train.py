from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd

print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1)

X = mnist.data
y = mnist.target.astype(int)

# ====== HIỂN THỊ 5 ẢNH MẪU ======
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
# Lấy 5 index random không trùng
random_indices = np.random.choice(len(X), 5, replace=False)

for ax, idx in zip(axes, random_indices):
    ax.imshow(X.iloc[idx].values.reshape(28, 28), cmap='gray')
    ax.set_title(f"Label: {y.iloc[idx]}")
    ax.axis('off')

plt.show()

# ====== KIỂM TRA PHÂN BỐ NHÃN ======
counter = Counter(y)
counter = counter.most_common()
counter_df = pd.DataFrame(counter, columns=['Số', 'Số lượng'])

print("\nPhân bố nhãn:")
print(counter_df)

# ====== CHIA DATA ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====== TEST NHIỀU K ======
k_values = range(1, 11)
accuracies = []

print("\nTesting K values...")

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    print(f"K = {k}, Accuracy = {acc}")

# ====== VẼ BIỂU ĐỒ ======
plt.figure()
plt.plot(k_values, accuracies, marker='o')
plt.title("Accuracy vs K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid()

plt.savefig("accuracy_vs_k.png")
print("Saved accuracy_vs_k.png")

# ====== CHỌN K TỐT NHẤT ======
best_k = k_values[accuracies.index(max(accuracies))]
print(f"Best K = {best_k}")

# ====== TRAIN LẠI ======
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ====== CONFUSION MATRIX ======
labels = list(range(10))
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("\nConfusion Matrix:")
print(cm)

# ====== VẼ CONFUSION MATRIX ======
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', values_format='d')

plt.title("Confusion Matrix (MNIST)")
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

plt.show()
