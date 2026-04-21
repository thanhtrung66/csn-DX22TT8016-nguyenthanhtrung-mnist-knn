# Nhận dạng chữ số viết tay bằng KNN

## 📌 Giới thiệu
Đề tài sử dụng thuật toán **K-Nearest Neighbors (KNN)** để phân loại chữ số viết tay từ bộ dữ liệu **MNIST**.

## ⚙️ Công cụ sử dụng<br>

**Ngôn ngữ lập trình**
- Python<br>

## 📦 Thư viện sử dụng
- `numpy`: xử lý dữ liệu số
- `scikit-learn`: xây dựng mô hình KNN
- `matplotlib`: vẽ biểu đồ
- `pandas`: xử lý, phân tích và quản lý dữ liệu dạng bảng
- `openpyxl`: giúp xuất dữ liệu ra file Excel

## 📂 Dataset: MNIST

- Số lượng mẫu: 70,000 ảnh
- Kích thước ảnh: 28x28 pixels
- Số lớp: 10 (từ 0 đến 9)
- Mỗi ảnh được biểu diễn thành vector 784 chiều

## 📌 Cách chạy trên CMD

* **Bước 1**: cài đặt thư viện<br>
Gõ lệnh `py -m pip install -r requirements.txt`<br>

📦 requirements.txt<br>
- `numpy`<br>
- `scikit-learn`<br>
- `matplotlib`<br>
- `pandas`<br>
- `openpyxl`<br>

* **Bước 2:** Chạy code<br>

_Trường hợp 1_: Chạy mô hình cơ bản<br>
`py knn_mnist.py`<br>
![Kết quả trường hợp 1](ket_qua_1.png)

_Trường hợp 2_: Tối ưu và phân tích tham số K, hiển thị ma trận nhầm lẫn<br>
`py train.py`<br>

### 📷 Hiển thị vài mẫu ảnh ngẫu nhiên
Dữ liệu MNIST gồm các ảnh chữ số viết tay kích thước 28x28 pixel.  
Để trực quan hóa dữ liệu, ta hiển thị ngẫu nhiên 5 ảnh từ dataset.<br>
![Mẫu ảnh](anh_ngau_nhien.png)

## 📊 Kiểm tra phân bố nhãn
![phân bố nhãn](phan_bo_nhan.png)

## 📊 Kết quả
- `Accuracy`: ~97%
- `Precision`: ~97%
- `Recall`: ~97%
- `F1-score`: ~97%

## 📈 Tối ưu tham số $K$
Biểu đồ dưới đây thể hiện độ chính xác theo các giá trị K khác nhau:
![Accuracy vs K](accuracy_vs_k.png)

_Kết quả cho thấy tuy K = 1 cho độ chính xác cao nhất, nhưng mô hình có thể nhạy với nhiễu (noise) và dễ bị overfitting. Trong thực tế, các giá trị K lớn hơn (như 3 hoặc 5) thường được cân nhắc để đảm bảo tính ổn định_.

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

_K nhỏ giúp mô hình phản ứng tốt với dữ liệu chi tiết, nhưng có thể dễ bị nhiễu_.

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

🎯 **Kết luận**<br>
* File: knn_mnist.py phù hợp để minh họa và học thuật toán<br>
* File: train.py phù hợp để làm đồ án và báo cáo chuyên sâu

## ⚠️ Hạn chế

- Tốn thời gian dự đoán với tập dữ liệu lớn
- Nhạy với dữ liệu nhiễu
- Không hiệu quả khi số chiều dữ liệu cao
