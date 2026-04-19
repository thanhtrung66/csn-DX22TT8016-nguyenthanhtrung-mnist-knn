# Import các thư viện cần thiết
from sklearn.datasets import fetch_openml              # Load dataset MNIST từ OpenML
from sklearn.model_selection import train_test_split   # Chia dữ liệu train/test
from sklearn.neighbors import KNeighborsClassifier     # Thuật toán KNN
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # Đánh giá model
import matplotlib.pyplot as plt                        # Vẽ biểu đồ
import numpy as np                                     # Xử lý mảng số
from collections import Counter                        # Đếm số lượng phần tử
import pandas as pd                                    # Xử lý dữ liệu dạng bảng

# ====== LOAD DATA ======
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1)  # Tải dataset MNIST (70,000 ảnh 28x28)

X = mnist.data                                # Dữ liệu ảnh (dạng vector 784 chiều)
y = mnist.target.astype(int)                  # Nhãn (0–9), chuyển về số nguyên

# ====== HIỂN THỊ 5 ẢNH MẪU ======
fig, axes = plt.subplots(1, 5, figsize=(10, 5))  # Tạo 1 hàng 5 cột để hiển thị ảnh

# Lấy 5 index ngẫu nhiên, không trùng
random_indices = np.random.choice(len(X), 5, replace=False)

# Lặp qua từng ảnh để hiển thị
for ax, idx in zip(axes, random_indices):
    ax.imshow(X.iloc[idx].values.reshape(28, 28), cmap='gray')  # Chuyển vector thành ảnh 28x28
    ax.set_title(f"Label: {y.iloc[idx]}")                       # Hiển thị nhãn
    ax.axis('off')                                              # Tắt trục tọa độ

plt.show()  # Hiển thị ảnh

# ====== KIỂM TRA PHÂN BỐ NHÃN ======
counter = Counter(y)                 # Đếm số lượng mỗi chữ số
counter = counter.most_common()      # Sắp xếp theo số lượng giảm dần
counter_df = pd.DataFrame(counter, columns=['Số', 'Số lượng'])  # Chuyển thành bảng

print("\nPhân bố nhãn:")
print(counter_df)  # In ra bảng phân bố

# ====== CHIA DATA ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42   # 80% train, 20% test
)

# ====== TEST NHIỀU K ======
k_values = range(1, 11)   # Thử K từ 1 đến 10
accuracies = []           # Lưu accuracy tương ứng

print("\nTesting K values...")

# Lặp qua từng giá trị K
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)  # Tạo model KNN với K hiện tại
    model.fit(X_train, y_train)                  # Huấn luyện model
    y_pred = model.predict(X_test)               # Dự đoán trên tập test
    
    acc = accuracy_score(y_test, y_pred)         # Tính độ chính xác
    accuracies.append(acc)                       # Lưu lại
    
    print(f"K = {k}, Accuracy = {acc}")          # In kết quả

# ====== VẼ BIỂU ĐỒ ======
plt.figure()
plt.plot(k_values, accuracies, marker='o')  # Vẽ đồ thị Accuracy theo K
plt.title("Accuracy vs K")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid()

plt.savefig("accuracy_vs_k.png")  # Lưu ảnh biểu đồ
print("Saved accuracy_vs_k.png")

# ====== CHỌN K TỐT NHẤT ======
best_k = k_values[accuracies.index(max(accuracies))]  # Lấy K có accuracy cao nhất
print(f"Best K = {best_k}")

# ====== train lại với K tốt nhất ======
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ====== confusion matrix ======
labels = list(range(10))  # Nhãn từ 0 đến 9
cm = confusion_matrix(y_test, y_pred, labels=labels)  # Tạo ma trận nhầm lẫn

print("\nConfusion Matrix:")
print(cm)

# ====== vẽ confusion matrix ======
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', values_format='d')  # Hiển thị ma trận

plt.title("Confusion Matrix (MNIST)")
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.savefig("confusion_matrix.png")  # Lưu ảnh
print("Saved confusion_matrix.png")

plt.show()  # Hiển thị kết quả
