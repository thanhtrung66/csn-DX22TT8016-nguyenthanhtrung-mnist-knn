from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1)

X = mnist.data
y = mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====== TEST NHIỀU K ======
k_values = range(1, 11)
accuracies = []

print("Testing K values...")

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

# ====== TRAIN LẠI VỚI K TỐT NHẤT ======
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ====== CONFUSION MATRIX ======
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

plt.show()