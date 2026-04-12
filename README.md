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

## 📷 Ma trận nhầm lẫn
Ma trận nhầm lẫn giúp đánh giá chi tiết khả năng phân loại của mô hình:
![Confusion Matrix](confusion_matrix.png)

K nhỏ giúp mô hình phản ứng tốt với dữ liệu chi tiết, nhưng có thể dễ bị nhiễu.

## 📌 Cách chạy
pip install -r requirements.txt  
python train.py
