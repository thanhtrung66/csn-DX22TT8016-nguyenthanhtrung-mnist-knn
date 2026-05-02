import os
# Khai báo biến đường dẫn chung để dễ quản lý và thay đổi
output_path = "thesis/abs"
# máy sẽ tạo thư mục đầu ra nếu chưa có
if not os.path.exists(output_path):
    os.makedirs(output_path)
# import các thư viện cần thiết
from sklearn.datasets import fetch_openml              # Load dataset MNIST từ OpenML
from sklearn.model_selection import train_test_split   # Chia dữ liệu train/test
from sklearn.neighbors import KNeighborsClassifier     # Thuật toán KNN
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # Đánh giá model
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score # các công cụ đo lường để đánh giá model
import seaborn as sns                            # Vẽ biểu đồ nhiệt
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

plt.draw() 
plt.savefig(os.path.join(output_path, "5random_images.png")) # lưu ảnh 5 số ngẫu nhiên vào thesis > abs
print(f"Saved 5random_images.png to {output_path}")

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

plt.savefig(os.path.join(output_path, "accuracy_vs_k.png"))  # Lưu ảnh biểu đồ vào tệp thesis > abs
print(f"Saved accuracy_vs_k.png to {output_path}")

# ====== CHỌN K TỐT NHẤT ======
best_k = k_values[accuracies.index(max(accuracies))]  # Lấy K có accuracy cao nhất
print(f"Best K = {best_k}")

# ====== HUẤN LUYỆN LẠI VỚI K TỐT NHẤT ======
print(f"\nTraining final model with Best K = {best_k}...")
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ====== TÍNH TOÁN CÁC CHỈ SỐ CHI TIẾT ======
# Tính Accuracy (Độ chính xác tổng quát)
accuracy = accuracy_score(y_test, y_pred)

# Tính Precision, Recall, F1-score (Trung bình cộng các lớp - macro average)
precision = precision_score(y_test, y_pred, average='macro') # Tính Precision trung bình cộng các lớp
recall = recall_score(y_test, y_pred, average='macro')       # Tính Recall trung bình cộng các lớp
f1 = f1_score(y_test, y_pred, average='macro')               # Tính F1-score trung bình cộng các lớp

# In kết quả ra màn hình Console
print("-" * 30)
print(f"Accuracy (Tổng quát): {accuracy:.4f}") 
print(f"Precision (Trung bình): {precision:.4f}")
print(f"Recall (Trung bình): {recall:.4f}")
print(f"F1-score (Trung bình): {f1:.4f}")
print("-" * 30)

# Xuất bảng báo cáo chi tiết cho từng chữ số (từ 0 đến 9)
print("\nBáo cáo chi tiết từng lớp (Classification Report):")
print(classification_report(y_test, y_pred))

# Lấy report dạng dict
report_dict = classification_report(y_test, y_pred, output_dict=True)

# Chuyển thành DataFrame
df_report = pd.DataFrame(report_dict).transpose()

# Xuất báo cáo ra Excel lưu vào tệp thesis > abs
df_report.to_excel(os.path.join(output_path, "classification_report.xlsx"), index=True) # xuất file excel các chỉ số vào thesis > abs
print("Đã xuất file Excel thành công!")

# ====== VẼ VÀ LƯU CONFUSION MATRIX ======
labels = list(range(10))
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', values_format='d')

plt.title(f"Confusion Matrix (MNIST) - Best K: {best_k}")
plt.xlabel("Predicted label")
plt.ylabel("True label")

plt.savefig(os.path.join(output_path, "confusion_matrix.png")) # Lưu ảnh ma trận nhầm lẫn thesis > abs
print(f"Saved confusion_matrix.png to {output_path}")  

# ====== VẼ HEATMAP (CONFUSION MATRIX) ======
# Tính toán ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)

# Tạo khung hình cho biểu đồ
plt.figure(figsize=(10, 8))

# Sử dụng seaborn để vẽ heatmap
# annot=True: Hiển thị số lượng bên trong các ô
# fmt='d': Định dạng số nguyên
# cmap='YlGnBu': Bộ màu Vàng - Xanh lá - Xanh dương (có thể đổi thành 'Blues' hoặc 'viridis')
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=range(10), yticklabels=range(10))

# Cấu hình tiêu đề và nhãn cho trục
plt.title(f"Heatmap Ma trận nhầm lẫn (MNIST) - Best K: {best_k}", fontsize=15)
plt.xlabel("Nhãn dự đoán (Predicted Label)", fontsize=12)
plt.ylabel("Nhãn thực tế (True Label)", fontsize=12)

# Lưu ảnh confusion matrix Heatmap vào thesis > abs
plt.savefig(os.path.join(output_path, "heatmap.png"))
print(f"Saved heatmap.png to {output_path}")

# Hiển thị tất cả biểu đồ
print("Generating and displaying visualizations...")
plt.show()
print("Kết thúc quy trình, vui lòng kiểm tra lại các kết quả dữ liệu trong tệp tương ứng.")

