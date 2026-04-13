# Nhận dạng chữ số viết tay bằng KNN

## 📌 Giới thiệu
Đề tài sử dụng thuật toán K-Nearest Neighbors (KNN) để phân loại chữ số viết tay từ bộ dữ liệu MNIST.

## ⚙️ Công nghệ sử dụng
- Python
- Scikit-learn
- NumPy
- Matplotlib

## 📊 Kết quả
- Accuracy: ~97%
- Precision: ~97%
- Recall: ~97%
- F1-score: ~97%

## 📈 Tối ưu tham số K
Biểu đồ dưới đây thể hiện độ chính xác theo các giá trị K khác nhau:
![Accuracy vs K](accuracy_vs_k.png)

Kết quả cho thấy giá trị K tối ưu là **K = 1** với độ chính xác cao nhất.

## 📈 Phân tích tham số K (Hyperparameter Tuning)

Trong đồ án này, mình đã thực hiện khảo sát giá trị $K$ từ 1 đến 10 để tìm ra điểm tối ưu cho mô hình.

- **Tại sao chọn phạm vi 1-10?**
    - **Đặc thù dữ liệu:** Với bộ dữ liệu MNIST, các đặc trưng đã được chuẩn hóa tốt, các láng giềng gần nhất thường mang đặc điểm rất giống nhau. Qua thực nghiệm, giá trị $K$ nhỏ (thường là số lẻ < 10) đem lại độ chính xác cao nhất (trên 96%).
    - **Hiệu năng:** Vì KNN là thuật toán "Lazy Learning", việc tăng $K$ quá lớn sẽ làm tăng khối lượng tính toán và thời gian dự đoán mà không cải thiện đáng kể độ chính xác.
    - **Tránh Underfitting:** Khi $K$ quá lớn, ranh giới phân loại giữa các chữ số bị làm mờ, dẫn đến mô hình bị đơn giản hóa quá mức.

*Kết quả khảo sát chi tiết có thể xem tại tệp `accuracy_vs_k.png`.*

## 📷 Ma trận nhầm lẫn
Ma trận nhầm lẫn giúp đánh giá chi tiết khả năng phân loại của mô hình:
![Confusion Matrix](confusion_matrix.png)

K nhỏ giúp mô hình phản ứng tốt với dữ liệu chi tiết, nhưng có thể dễ bị nhiễu.

## 📌 Cách chạy
Bước 1: cài đặt thư viện<br>
py -m pip install -r requirements.txt<br>
Bước 2: Chạy code<br>
Trường hợp 1: py knn_mnist.py<br>
Trường hợp 2: py train.py<br>

## 📦 Thư viện sử dụng
- numpy: xử lý dữ liệu số
- scikit-learn: xây dựng mô hình KNN
- matplotlib: vẽ biểu đồ
