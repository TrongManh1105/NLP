# 📰 English Clickbait Headline Detection using DistilBERT

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-DistilBERT-yellow.svg)](https://huggingface.co/distilbert-base-uncased)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg?logo=gradio&logoColor=white)](https://gradio.app/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.13%25-brightgreen.svg)]()
[![F1-Score](https://img.shields.io/badge/F1--Score-0.9913-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Dự án nghiên cứu và xây dựng mô hình học sâu (**Deep Learning / NLP**) nhằm **tự động phát hiện và phân loại tiêu đề giật gân (Clickbait Detection)** trên các nền tảng báo chí điện tử và mạng xã hội, ứng dụng mô hình ngôn ngữ tiền huấn luyện **DistilBERT** kết hợp kỹ thuật giải thích mô hình **Explainable AI (XAI)** và giao diện thử nghiệm tương tác trực quan với **Gradio**.

---

## 📌 Mục lục

- [Giới thiệu đề tài](#-giới-thiệu-đề-tài)
- [Tính năng nổi bật](#-tính-năng-nổi-bật)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Bộ dữ liệu (Dataset)](#-bộ-dữ-liệu-dataset)
- [Huấn luyện mô hình (Model Training)](#-huấn-luyện-mô-hình-model-training)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Giải thích mô hình (Explainable AI - XAI)](#-giải-thích-mô-hình-explainable-ai---xai)
- [Giao diện thử nghiệm (Gradio Web UI)](#-giao-diện-thử-nghiệm-gradio-web-ui)
- [Cấu trúc mã nguồn](#-cấu-trúc-mã-nguồn)
- [Hướng dẫn cài đặt & sử dụng](#-hướng-dẫn-cài-đặt--sử-dụng)
- [Thông tin tác giả & Người hướng dẫn](#-thông-tin-tác-giả--người-hướng-dẫn)

---

## 📖 Giới thiệu đề tài

Trong kỷ nguyên truyền thông số, hàng triệu bài báo và nội dung số được xuất bản mỗi ngày. Nhằm tối ưu hóa số lượng truy cập và tương tác, nhiều nhà sáng tạo nội dung đã sử dụng các tiêu đề giật gân (**Clickbait**) nhằm kích thích trí tò mò hoặc tạo cảm xúc quá mức. Mặc dù làm gia tăng lượt nhấp chuột trong ngắn hạn, việc lạm dụng Clickbait dẫn đến trải nghiệm tiêu cực, gây nhiễu loạn luồng thông tin và làm suy giảm uy tín của các kênh truyền thông.

Dự án này giải quyết bài toán phân loại nhị phân văn bản:
$$\text{Input: Headline (Text)} \longrightarrow \text{Output: } \{0: \text{Non-clickbait}, 1: \text{Clickbait}\}$$

Bằng cách áp dụng mô hình **DistilBERT** (kỹ thuật *Knowledge Distillation* kế thừa từ BERT), hệ thống đạt độ chính xác vượt trội lên tới **99.13%**, đồng thời giảm 40% dung lượng và tăng 60% tốc độ suy luận so với mô hình BERT gốc.

---

## ✨ Tính năng nổi bật

- ⚡ **Hiệu năng xuất sắc**: Đạt độ chính xác kiểm thử **99.13%** và F1-Score **0.9913** trên tập dữ liệu chuẩn 32,000 tiêu đề.
- 🎯 **Kiến trúc tối ưu**: Sử dụng `distilbert-base-uncased` với cơ chế Self-Attention giúp nắm bắt trọn vẹn ngữ cảnh ngữ nghĩa 2 chiều của câu.
- 🔍 **Khả năng giải thích (Explainable AI - XAI)**: Tích hợp **LIME** (`LimeTextExplainer`) và **transformers-interpret** giúp bóc tách trọng số, xác định chính xác từ ngữ nào khiến mô hình phân loại tiêu đề là clickbait.
- 🖥️ **Giao diện tương tác trực quan**: Cung cấp giao diện Web bằng **Gradio** cho phép người dùng kiểm tra tiêu đề theo thời gian thực (Real-time Inference), hiển thị xác suất phân bố và trực quan hóa token importance.

---

## 🏗️ Kiến trúc hệ thống

```mermaid
flowchart TD
    A[Raw Dataset: clickbait_data.csv\n32,000 Headlines] --> B[Data Preprocessing & EDA\nToken Length, WordCloud, Class Distribution]
    B --> C[Stratified Split\n80% Train | 10% Valid | 10% Test]
    C --> D[DistilBERT Tokenizer\nWordPiece Tokenization, Attention Mask]
    D --> E[Fine-tuning DistilBERT\nLearning Rate: 2e-5, Batch: 16, Epochs: 3]
    E --> F[Model Evaluation\nAccuracy: 99.13%, F1: 99.13%, Confusion Matrix]
    F --> G[Explainable AI - XAI\nLIME & transformers-interpret Token Attribution]
    F --> H[Interactive Demo\nGradio Web Interface]
```

---

## 📊 Bộ dữ liệu (Dataset)

Dự án sử dụng bộ dữ liệu `clickbait_data.csv` bao gồm **32,000** dòng tiêu đề tiếng Anh được gán nhãn cân bằng hoàn hảo:

| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Tổng số mẫu** | 32,000 headlines |
| **Số cột** | 2 cột (`headline`, `clickbait`) |
| **Phân bố nhãn** | **50% Clickbait** (16,000 mẫu) & **50% Non-clickbait** (16,000 mẫu) |
| **Tập Train (80%)** | 25,600 mẫu |
| **Tập Validation (10%)** | 3,200 mẫu |
| **Tập Test (10%)** | 3,200 mẫu |
| **Chiến lược chia** | `train_test_split` với `stratify=df['clickbait']` và `random_state=42` |

### Ví dụ dữ liệu:
- **Clickbait (1)**:
  - *"Should I Get Bings"*
  - *"Which TV Female Friend Group Do You Belong In"*
  - *"The New 'Star Wars: The Force Awakens' Trailer Is Here To Give You Chills"*
  - *"You Won't Believe How This New OpenAI Model Passed the Bar Exam with a Perfect Score"*
- **Non-clickbait (0)**:
  - *"Scientists Discover New Species in Amazon Rainforest"*
  - *"Government Releases Annual Economic Report"*
  - *"Inside the Secret AI Algorithm That Boosted Amazon's Sales by 40% Overnight"*

---

## ⚙️ Huấn luyện mô hình (Model Training)

Mô hình được huấn luyện bằng Hugging Face `Trainer` API trên môi trường GPU:

```python
training_args = TrainingArguments(
    output_dir="./distilbert_clickbait_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

- **Mô hình gốc**: `distilbert-base-uncased` (6 Transformer layers, 12 attention heads, 768 hidden dimension, ~66M parameters).
- **Hàm tối ưu**: AdamW kết hợp Linear Learning Rate Scheduler.
- **Hàm mất mát**: Cross-Entropy Loss.
- **Tiêu chí chọn checkpoint tốt nhất**: `eval_f1`.

---

## 📈 Kết quả thực nghiệm

### 1. Tổng quan các chỉ số trên tập Test (3,200 mẫu)

| Chỉ số (Metric) | Kết quả (%) |
| :--- | :---: |
| **Test Accuracy** | **99.13%** |
| **Test Precision** | **99.00%** |
| **Test Recall** | **99.25%** |
| **Test F1-Score** | **99.13%** |
| **Evaluation Loss** | **0.0572** |

### 2. Chi tiết Classification Report

| Phân lớp (Class) | Precision | Recall | F1-Score | Số mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Non-clickbait (0)** | 0.99 | 0.99 | 0.99 | 1,600 |
| **Clickbait (1)** | 0.99 | 0.99 | 0.99 | 1,600 |
| **Accuracy** | | | **0.99** | 3,200 |
| **Macro Avg** | 0.99 | 0.99 | 0.99 | 3,200 |
| **Weighted Avg** | 0.99 | 0.99 | 0.99 | 3,200 |

### 3. Phân tích ma trận nhầm lẫn (Confusion Matrix)

Trong tổng số 3,200 mẫu kiểm thử:
- **Dự đoán chính xác**: **3,172 mẫu** (99.13%)
- **Dự đoán sai**: **28 mẫu** (0.87%)
  - Non-clickbait dự đoán nhầm thành Clickbait: 16 mẫu / 1,600.
  - Clickbait dự đoán nhầm thành Non-clickbait: 12 mẫu / 1,600.

Tỷ lệ lỗi phân bổ đồng đều ở cả 2 lớp, chứng minh mô hình không bị thiên vị (unbiased) và có khả năng tổng quát hóa rất cao trên dữ liệu thực tế.

---

## 🧠 Giải thích mô hình (Explainable AI - XAI)

Để tránh hiện tượng mô hình là một chiếc "hộp đen" (Black-box), dự án áp dụng:
1. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Xây dựng mô hình đại diện cục bộ xung quanh điểm dự đoán.
   - Định lượng mức độ đóng góp (impact score) của từng từ vào quyết định phân loại.
2. **transformers-interpret (SequenceClassificationExplainer)**:
   - Khai thác trực tiếp cơ chế Attention và đạo hàm Gradient của DistilBERT.
   - Trực quan hóa các token quan trọng nhất.

> **Quan sát**: Các cụm từ đóng góp tích cực nhất vào nhãn Clickbait thường là đại từ ngôi thứ hai (*"You"*, *"Your"*), từ chỉ mức độ ngạc nhiên (*"Unbelievable"*, *"Shock"*, *"Secret"*), cụm kích thích tò mò (*"Won't Believe"*, *"What Happened Next"*), hoặc dạng số đếm danh sách (*"10 Things"*, *"5 Ways"*).

---

## 💻 Giao diện thử nghiệm (Gradio Web UI)

Hai phiên bản giao diện người dùng tương tác được phát triển trong dự án:
- **`Demo_Clickbait.ipynb`**: Giao diện tiêu chuẩn với chủ đề Soft theme, hỗ trợ kiểm tra nhanh tiêu đề, hiển thị nhãn dự đoán kèm thanh đo xác suất tin cậy (Confidence Bar).
- **`Demonew.ipynb`**: Giao diện mở rộng hiển thị quy trình tokenization, trọng số đóng góp của từng token và trực quan hóa chi tiết đầu ra của mô hình.

---

## 📁 Cấu trúc mã nguồn

```bash
NLP/
├── clickbait_data.csv       # Bộ dữ liệu 32,000 tiêu đề tiếng Anh gán nhãn 0/1
├── NLP.ipynb                # Pipeline hoàn chỉnh: EDA, Preprocessing, Fine-tuning & Evaluation, LIME & XAI
├── Demo_Clickbait.ipynb     # Giao diện Gradio tương tác phân loại tiêu đề thời gian thực
├── Demonew.ipynb            # Giao diện Gradio nâng cao trực quan hóa tokens & kết quả phân tích
└── README.md                # Tài liệu mô tả dự án
```

---

## 🚀 Hướng dẫn cài đặt & sử dụng

### 1. Clone repository
```bash
git clone https://github.com/TrongManh1105/NLP.git
cd NLP
```

### 2. Cài đặt các thư viện phụ thuộc
```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn wordcloud lime transformers-interpret gradio captum
```

### 3. Huấn luyện và chạy thử nghiệm
- **Chạy notebook huấn luyện**:
  Mở và thực thi toàn bộ notebook [NLP.ipynb](NLP.ipynb) trên Jupyter Notebook hoặc Google Colab (khuyến nghị sử dụng GPU T4).
- **Chạy giao diện Gradio**:
  Mở [Demo_Clickbait.ipynb](Demo_Clickbait.ipynb) hoặc [Demonew.ipynb](Demonew.ipynb) và chạy các ô lệnh để khởi chạy server Gradio cục bộ hoặc nhận liên kết chia sẻ công khai (`share=True`).

---

## 👨‍🎓 Thông tin tác giả & Người hướng dẫn

- **Sinh viên thực hiện**: **Nguyễn Trọng Mạnh Tiến**
- **Mã sinh viên**: `2351260689`
- **Lớp**: `65TTNT`
- **Khoa**: Công nghệ thông tin
- **Trường**: Đại học Thủy Lợi (Thuyloi University)
- **Giảng viên hướng dẫn**: **PGS.TS. Nguyễn Quang Hoan**
- **Học phần**: Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP)

---

## 📜 Giấy phép (License)

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
Mọi đóng góp, phản hồi hoặc câu hỏi liên quan xin vui lòng mở [Issue](https://github.com/TrongManh1105/NLP/issues) trên GitHub.
