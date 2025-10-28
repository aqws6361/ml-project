# AI 圖像生成與分析工具

這是一個基於 PyTorch 的命令列工具，整合了 Hugging Face 上的 CLIP 和 擴散 (Diffusion) 模型，提供兩種主要的 AI 圖像處理功能：

1.  **零樣本圖像分類 (Zero-Shot Classification)**: 使用 CLIP-based 模型來判斷一張圖片最符合哪一個文字標籤。
2.  **文字生成圖像 (Text-to-Image Generation)**: 使用 Diffusion-based 模型（如 Stable Diffusion）根據文字提示生成圖像。

## 🚀 功能

* **CLIP 零樣本分類**: 給定一張圖片和多個候選標籤，模型會計算圖片與哪個標籤的關聯性最高。
* **擴散模型圖像生成**: 根據您提供的文字提示 (prompt) 和負面提示 (negative prompt) 來創造新的圖像。
* **CUDA 加速**: 自動檢測並使用 NVIDIA GPU (CUDA) 進行快速推理，若無 GPU 則會退回至 CPU 執行（並顯示警告）。
* **模型可自訂**: 允許用戶透過參數指定要從 Hugging Face 下載的特定 CLIP 或擴散模型。

## ⚙️ 安裝與設定

1.  **複製專案庫 (Clone)**:
    ```bash
    git clone [https://github.com/aqws6361/ml-project.git](https://github.com/aqws6361/ml-project.git)
    cd ml-project
    ```

2.  **安裝依賴套件**:
    建議在 Python 虛擬環境中安裝。
    ```bash
    # 建立 requirements.txt 檔案 (內容請參考上方)
    pip install -r requirements.txt
    ```
    *或者，手動安裝：*
    ```bash
    pip install torch transformers diffusers accelerate Pillow
    ```

## 🤖 如何使用

此腳本透過 `main.py` 運行，並使用 `--demo` 參數來選擇模式。

### 1. 範例：CLIP 零樣本分類

此模式需要一張本地圖片 (`--image_path`) 和至少一個標籤 (`--labels`)。

**範例指令：**
```bash
python main.py --demo clip \
    --image_path "./path/to/your/image.jpg" \
    --labels "a photo of a dog" "a photo of a cat" "a photo of a bird"