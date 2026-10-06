# Chạy SPFT trên bộ dữ liệu WeThink

README này mô tả nhanh cách cài môi trường, chuẩn bị dataset và chạy sweep SPFT.

## 1. Cài đặt môi trường

Chạy các lệnh sau từ thư mục gốc của project:

```bash
conda create --prefix ./env python=3.12 -y
conda activate ./env

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -U "huggingface_hub[cli]"
```

Cần cài sẵn `conda`, `bash` và `unzip`. Nếu Hugging Face yêu cầu đăng nhập, chạy:

```bash
hf auth login
```

## 2. Chuẩn bị dataset

```bash
bash scripts/prepare_wethink.sh
```

> Lưu ý: bước này sẽ tải dataset, tải ảnh và giải nén dữ liệu nên có thể chạy khá lâu. Thầy đừng bấm `Ctrl+C` quá sớm.

## 3. Chạy sweep SPFT

Lệnh dưới đây chạy lần lượt với `learning rate` bằng `6e-5` và `7e-5`, đồng thời cố định `spft_lambda = 0.2`:

```bash
for lr in 6e-5 7e-5; do
  bash scripts/train_wethink_spft.sh \
    --learning_rate "$lr" \
    --spft_lambda 0.2 \
    --output_dir "saves/qwen2_5vl-3b/wethink_spft_lr_${lr}_lambda_0.2"
done
```

## 4. Thư mục đầu ra

Sau khi train, kết quả sẽ nằm trong hai thư mục tương ứng:

```text
saves/qwen2_5vl-3b/wethink_spft_lr_6e-5_lambda_0.2/
saves/qwen2_5vl-3b/wethink_spft_lr_7e-5_lambda_0.2/
```

Mỗi thư mục chứa checkpoint và log của một lần chạy.
