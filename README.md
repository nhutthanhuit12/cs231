# CS231 - Nhập môn Thị giác máy tính (Computer Vision)
## Đồ án: Phân loại hoa (Flower Classification) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

> **Lưu ý:** Dự án này là một phần của môn học CS231 tại Trường Đại học Công nghệ Thông tin (UIT).

## 📝 Mục lục
1. [Giới thiệu](#-giới-thiệu)
2. [Thành viên nhóm](#-thành-viên-nhóm)
3. [Dữ liệu (Dataset)](#-dữ-liệu-dataset)
4. [Phương pháp (Methodology)](#-phương-pháp-methodology)
6. [Kết quả (Results)](#-kết-quả-results)
5. [Cài đặt & Hướng dẫn sử dụng](#-cài-đặt--hướng-dẫn-sử-dụng)
7. [Tham khảo](#-tham-khảo)

---

## 📖 Giới thiệu
Đồ án này tập trung vào việc sử dụng các phương pháp trích xuất đặc trưng kết hợp vớ mô hình máy học SVM để phân loại loài hoa

## 👥 Thành viên nhóm

| STT | MSSV | Họ và tên | Github |
|:---:|:---:|:---|:---|
| 1 | 23521451 | Nguyễn Nhựt Thành | https://github.com/nhutthanhuit12 |

## 📊 Dữ liệu (Dataset)
Nhóm sử dụng bộ dữ liệu **Oxford 102 Flowers Dataset**

- **Số lượng:** 102 loại hoa, tổng cộng hơn 8.000 hình ảnh.
- **Cấu trúc thư mục:**
```text
  data/
  ├── train/
  │   ├── class_1/
  │   └── ...
  ├── val/
  └── test/
```

## 🛠 Phương pháp (Methodology)
Dự án thực hiện các thí nghiệm trên nhiều phương pháp tiếp cận:

1.  Trích xuất đặc trưng: SIFT, HOG, HIST, ResNet50
2.  Mô hình máy học: SVM
3.  Deploy: Streamlit

## 📈Kết quả (Results)
| Method | Accuracy | F1-Score (Weighted) |
| :--- | :---: | :---: |
| **RESNET50** | 0.925428 | 0.924075 |
| **SIFT** | 0.596577 | 0.587918 |
| **HIST** | 0.422983 | 0.413607 |
| **HOG** | 0.249389 | 0.242226 |

## ⚙ Cài đặt & Hướng dẫn sử dụng
Bước 1: Clone dự án
```bash
git clone [https://github.com/nhutthanhuit12/cs231.git](https://github.com/nhutthanhuit12/cs231.git)
cd cs231
```
Bước 2: Tải thư viện cần thiết
```bash
pip install -r requirements.txt
```
Bước 3: Chạy demo
```bash
streamlit run app.py
```

## 📚 Tham khảo
* **Dataset:** [Oxford 102 Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
* **ResNet50:** [Deep Residual Learning for Image Recognition (ResNet Paper)](https://arxiv.org/abs/1512.03385)
* **SVM:** [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
* **Deploy:** [Streamlit Documentation](https://docs.streamlit.io/)
