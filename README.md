# Nhận dạng chữ số viết tay bằng KNN

## 📌 Giới thiệu
Đề tài sử dụng thuật toán **K-Nearest Neighbors (KNN)** để phân loại chữ số viết tay từ bộ dữ liệu **MNIST**.

## ⚙️ Ngôn ngữ lập trình và thư viện sử dụng<br>

**Ngôn ngữ lập trình**
- Python<br>

## 📦 Thư viện sử dụng
- `numpy`: xử lý dữ liệu số
- `scikit-learn`: xây dựng mô hình KNN
- `matplotlib`: vẽ biểu đồ
- `pandas`: xử lý, phân tích và quản lý dữ liệu dạng bảng
-  bộ dữ liệu **MNIST**

## 📊 Kết quả
- `Accuracy`: ~97%
- `Precision`: ~97%
- `Recall`: ~97%
- `F1-score`: ~97%

## 📈 Tối ưu tham số $K$
Biểu đồ dưới đây thể hiện độ chính xác theo các giá trị K khác nhau:
![Accuracy vs K](accuracy_vs_k.png)

Kết quả cho thấy giá trị $K$ tối ưu là **K = 1** với độ chính xác cao nhất.

## 📈 Phân tích tham số $K$ (Hyperparameter Tuning)

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

## 📌 Cách chạy trên CMD

* **Bước 1**: cài đặt thư viện<br>
Gõ lệnh `py -m pip install -r requirements.txt`<br>

* **Bước 2:** Chạy code<br>

*_Trường hợp 1_: `py knn_mnist.py`<br>

*_Trường hợp 2_: `py train.py`<br>

📌 Chi tiết từng file

🔹 knn_mnist.py
Sử dụng giá trị K cố định (K = 3) để huấn luyện mô hình
Tính đầy đủ các chỉ số đánh giá:
`Accuracy`
`Precision`
`Recall`
`F1-score`
Hiển thị ma trận nhầm lẫn dưới dạng hình ảnh đơn giản

**Ưu điểm:**

Code ngắn gọn, dễ hiểu
Phù hợp để minh họa thuật toán KNN cơ bản

**Nhược điểm:**

Không kiểm tra nhiều giá trị $K$ nên chưa tối ưu mô hình
Confusion Matrix chưa hiển thị rõ nhãn

🔹 train.py

Thử nghiệm nhiều giá trị K từ 1 đến 10
Vẽ biểu đồ Accuracy vs K để lựa chọn K tối ưu
Tự động chọn _K_ tốt nhất và huấn luyện lại mô hình
Hiển thị Confusion Matrix với đầy đủ nhãn (0–9)
Lưu các biểu đồ dưới dạng file `.png`

**Ưu điểm:**

Có quy trình tối ưu hyperparameter rõ ràng
Trực quan hóa dữ liệu tốt, phù hợp cho báo cáo
Confusion Matrix hiển thị chuyên nghiệp

**Nhược điểm:**

Chưa tính các chỉ số như `Precision`, `Recall`, `F1-score`
Thời gian chạy lâu hơn do thử nhiều giá trị _K_

🎯 **Kết luận**
File: knn_mnist.py phù hợp để minh họa và học thuật toán<br>
File: train.py phù hợp để làm đồ án và báo cáo chuyên sâu

Trong thực tế, nên kết hợp ưu điểm của cả hai file:

Đâu tiên dùng train.py để tìm _K_ tối ưu, sau đó bổ sung thêm `Precision`, `Recall`, `F1-score` để đánh giá toàn diện mô hình.
