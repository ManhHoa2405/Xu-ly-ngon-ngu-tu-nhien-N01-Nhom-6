# Hệ thống Tìm kiếm văn bản & Phân loại tin tức tiếng Việt

**Bài tập lớn học phần:** Xử lý ngôn ngữ tự nhiên (Natural Language Processing — NLP)

**Trường:** Đại học Phenikaa

**Nhóm thực hiện:** `[Điền Mã nhóm / Tên nhóm]`

**Thành viên:**

* `[Họ và tên 1]` — `[MSSV 1]` — `[Phụ trách]`

* `[Họ và tên 2]` — `[MSSV 2]` — `[Phụ trách]`

* `[Họ và tên 3]` — `[MSSV 3]` — `[Phụ trách]`

## 1. Mục tiêu dự án

Dự án xây dựng một quy trình (pipeline) Xử lý ngôn ngữ tự nhiên hoàn chỉnh, có thể tái lập từ dữ liệu thô đến đánh giá và phân tích lỗi, giải quyết đồng thời hai bài toán trên cùng một kho tin tức tiếng Việt:

1. **Phân loại tin tức tiếng Việt (Text Classification):** Tự động gán nhãn chuyên mục (`Cong_nghe`, `The_thao`, `Kinh_te`, `Giao_duc`, `Suc_khoe`) cho bài báo mới. Thực hiện khảo sát đối chứng có kiểm soát (`E1`–`E5`) giữa:

   * Tách theo âm tiết vs. Tách từ ghép tiếng Việt (`underthesea`).

   * Biểu diễn `CountVectorizer` (Bag-of-Words) vs. `TfidfVectorizer`.

   * Unigram `(1,1)` vs. Unigram + Bigram `(1,2)`.

   * Mô hình cơ sở **Multinomial Naive Bayes** vs. Mô hình cải tiến **Logistic Regression**.

2. **Hệ thống Tìm kiếm văn bản (Mini Search Engine — Information Retrieval):** Xếp hạng và trả về Top-$k$ bài báo liên quan nhất với câu truy vấn của người dùng dựa trên mô hình không gian vector (Vector Space Model) và độ tương đồng **Cosine Similarity**, đối chứng qua 3 cấu hình (`S1`–`S3`).

## 2. Nguồn dữ liệu và nguyên tắc thu thập

* **Nguồn:** Báo điện tử công khai [VnExpress](https://vnexpress.net/) thuộc 5 chuyên mục:

  * Công nghệ (`https://vnexpress.net/so-hoa`)

  * Thể thao (`https://vnexpress.net/the-thao`)

  * Kinh tế (`https://vnexpress.net/kinh-doanh`)

  * Giáo dục (`https://vnexpress.net/giao-duc`)

  * Sức khỏe (`https://vnexpress.net/suc-khoe`)

* **Đơn vị mẫu:** 01 bài báo gồm tiêu đề (`title`), mô tả ngắn (`description`), nội dung chính (`content`), đường dẫn (`url`), nhãn chuyên mục (`category`) và thời gian thu thập (`crawled_at`).

* **Quy tắc lọc:** Loại bỏ các bài thiếu tiêu đề, nội dung dưới 30 từ và lọc trùng lặp theo `url` và `title`.

* **Đạo đức dữ liệu & Ẩn thông tin cá nhân (PII Masking):**

  * Dữ liệu chỉ phục vụ mục đích học tập và nghiên cứu trong khuôn khổ học phần.

  * Có độ trễ (`time.sleep`) giữa các yêu cầu thu thập để tôn trọng máy chủ nguồn.

  * Toàn bộ đường dẫn web (`URL`), địa chỉ email (`Email`) và số điện thoại Việt Nam (`Phone`) xuất hiện trong văn bản được tự động phát hiện và ẩn bằng biểu thức chính quy (Regex) trước khi đưa vào huấn luyện.

## 3. Cấu trúc thư mục

```
.
├── NLP_News_Search_and_Classification-v2.ipynb   # Notebook chính chứa toàn bộ pipeline (Bước 1–6)
├── NLP_BTL_MaNhom.pdf                            # Báo cáo bài tập lớn (20–30 trang)
├── README.md                                     # Hướng dẫn dự án và tái lập kết quả
├── data/                                         # Dữ liệu trung gian (tự động tạo khi chạy)
│   ├── vnexpress_raw.csv                         # Dữ liệu bài báo thô sau khi thu thập
│   └── vnexpress_processed.csv                   # Dữ liệu sau tiền xử lý (âm tiết & từ ghép)
└── results/                                      # Bảng số liệu & biểu đồ (tự động xuất khi chạy)
    ├── environment_versions.json                 # Phiên bản Python & các thư viện sử dụng
    ├── eda_corpus_overview.csv                   # Thống kê tổng quan (tài liệu, câu, âm tiết)
    ├── eda_summary.csv                           # Thống kê chi tiết theo từng chuyên mục
    ├── eda_overview.png                          # Biểu đồ phân bố lớp, số câu và độ dài bài
    ├── eda_vocab_comparison.csv                  # So sánh Vocabulary trước/sau tiền xử lý
    ├── data_split_stats.csv                      # Thống kê chia tập Train / Val / Test
    ├── val_hyperparameter_tuning.csv             # Kết quả chọn siêu tham số trên Validation
    ├── classification_comparison.csv             # Bảng đối chứng phân loại E1–E5 (Test)
    ├── classification_comparison.png             # Biểu đồ so sánh Accuracy & Macro-F1 (E1–E5)
    ├── confusion_matrix_e1_vs_e5.png             # Ma trận nhầm lẫn so sánh E1 vs E5
    ├── top_features_per_class.csv                # Top 10 từ khóa trọng số cao nhất mỗi lớp
    ├── classification_error_analysis.csv         # Top 10 ví dụ phân loại sai (Error Analysis)
    ├── search_comparison.csv                     # Bảng đối chứng tìm kiếm S1–S3 (P@5, MRR@5)
    ├── search_top3_examples.csv                  # Minh họa Top-3 bài báo trả về của S3
    └── search_error_analysis.csv                 # Phân tích ca khó tìm kiếm (OOV, đồng nghĩa, đa nghĩa)

```

## 4. Môi trường và cài đặt

* **Môi trường khuyến nghị:** Python 3.10+ trên **Google Colab** hoặc **Jupyter Notebook**.

* **Hạt giống ngẫu nhiên (Random Seed):** Cố định `RANDOM_STATE = 42` trong toàn bộ quá trình chia dữ liệu và huấn luyện mô hình.

* **Các thư viện chính:**

  * Thu thập & phân tích HTML: `requests`, `beautifulsoup4`, `lxml`

  * Xử lý ngôn ngữ tiếng Việt: `underthesea` (`sent_tokenize`, `word_tokenize`)

  * Học máy & đánh giá: `scikit-learn`, `pandas`, `numpy`

  * Trực quan hóa: `matplotlib`, `seaborn`

Nếu chạy trên máy cục bộ (Local), cài đặt thư viện bằng lệnh:

```
pip install requests beautifulsoup4 lxml underthesea scikit-learn pandas numpy matplotlib seaborn

```

*(Trên Google Colab, lệnh cài đặt đã được tích hợp sẵn ở ô code đầu tiên của Notebook. Phiên bản chính xác của từng thư viện sau khi chạy sẽ được lưu tự động tại `results/environment_versions.json`.)*

## 5. Hướng dẫn chạy từ đầu

### Cách 1: Chạy trên Google Colab (Khuyến nghị)

1. Tải file `NLP_News_Search_and_Classification-v2.ipynb` lên [Google Colab](https://colab.research.google.com/).

2. *(Tùy chọn)* Tại **Bước 1**, điều chỉnh số lượng bài thu thập:

   * **Chạy thử nhanh (\~2–3 phút):** đặt `MAX_PAGES_PER_CAT = 3`, `MAX_ARTICLES_PER_CAT = 60`.

   * **Chạy lấy số liệu báo cáo chính thức:** giữ nguyên `MAX_PAGES_PER_CAT = 10`, `MAX_ARTICLES_PER_CAT = 300` (hoặc tăng lên `15` / `400`).

3. Trên thanh menu, chọn **Runtime (Thời gian chạy) → Run all (Chạy tất cả)**.

4. Sau khi hoàn tất, kiểm tra dữ liệu trung gian trong thư mục `data/` và toàn bộ bảng biểu, hình ảnh trong thư mục `results/`.

### Cách 2: Chạy lại trên dữ liệu đã thu thập sẵn (để tái tạo chính xác 100% số liệu)

* Vì tin tức trực tuyến thay đổi theo thời gian, để tái tạo **chính xác tuyệt đối** các bảng số liệu trong báo cáo, hãy đặt file `vnexpress_raw.csv` (đã nộp kèm) vào thư mục `data/vnexpress_raw.csv` trước khi nhấn **Run all**.

* Notebook đã tích hợp cơ chế tự động kiểm tra: nếu thấy file `data/vnexpress_raw.csv`, hệ thống sẽ bỏ qua bước thu thập mới và chạy thẳng từ Bước 1 (EDA) đến Bước 6 trên đúng tập dữ liệu đó.

## 6. Cách tái tạo các bảng số liệu và đồ thị trong báo cáo

| Hạng mục trong báo cáo | File kết quả đầu ra | Vị trí tạo trong Notebook | 
| ----- | ----- | ----- | 
| Thông tin môi trường & phiên bản thư viện | `results/environment_versions.json` | Cell cài đặt & khởi tạo | 
| Thống kê tổng quan corpus & theo chuyên mục | `results/eda_corpus_overview.csv`, `results/eda_summary.csv` | Bước 1 — EDA | 
| Biểu đồ phân bố lớp, số câu & độ dài bài | `results/eda_overview.png` | Bước 1 — EDA | 
| So sánh kích thước Vocabulary trước/sau xử lý | `results/eda_vocab_comparison.csv` | Bước 2 — Tiền xử lý | 
| Bảng chia tập Train / Validation / Test (70/15/15) | `results/data_split_stats.csv` | Bước 3 — Chia dữ liệu | 
| Kết quả tinh chỉnh tham số (`alpha`, `C`) trên Val | `results/val_hyperparameter_tuning.csv` | Bước 4.1 — Tuning | 
| Bảng & biểu đồ đối chứng phân loại (`E1`–`E5`) | `results/classification_comparison.csv`, `.png` | Bước 4.2 — Thực nghiệm A | 
| Bảng đối chứng tìm kiếm (`S1`–`S3`) & ví dụ Top-3 | `results/search_comparison.csv`, `search_top3_examples.csv` | Bước 4 & 5 — Thực nghiệm B | 
| Ma trận nhầm lẫn so sánh Baseline `E1` vs `E5` | `results/confusion_matrix_e1_vs_e5.png` | Bước 6.1 — Đánh giá | 
| Top 10 từ khóa đặc trưng theo từng chuyên mục | `results/top_features_per_class.csv` | Bước 6.2 — Giải thích mô hình | 
| Phân tích 10 lỗi phân loại & ca khó tìm kiếm | `results/classification_error_analysis.csv`, `search_error_analysis.csv` | Bước 6.3 — Phân tích lỗi | 
| Demo trực tiếp (Phân loại + Tìm kiếm Top-5) | Đầu ra trực tiếp tại hàm `demo_system(...)` | Bước 6.4 — Demo | 

## 7. Tài liệu tham khảo chính

1. Daniel Jurafsky & James H. Martin, *Speech and Language Processing* (3rd ed. draft), Chapters 2, 4, 5, 6.

2. Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze, *Introduction to Information Retrieval* — Vector Space Model & TF-IDF.

3. Slide bài giảng học phần Xử lý ngôn ngữ tự nhiên (Module 3, 4, 7 & Lab 1&2), PGS.TS. Phạm Tiến Lâm, Trường Đại học Phenikaa.

4. Tài liệu thư viện [scikit-learn](https://scikit-learn.org/) (`Pipeline`, `CountVectorizer`, `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`) và [underthesea](https://github.com/undertheseanlp/underthesea).