# AI 圖像生成專案 (Text-to-Image AI Project)

本專案展示了如何使用 CLIP 進行零樣本圖像分類，以及如何使用擴散模型 (Stable Diffusion) 根據文字提示生成圖像。

## 專案結構

- `src/`: 包含主要的 Python 原始碼。
  - `clip_utils.py`: CLIP 相關功能。
  - `diffusion_utils.py`: 擴散模型相關功能。
  - `main.py`: 主執行腳本，可選擇執行 CLIP 或擴散模型演示。
- `images/`: 存放測試 CLIP 功能的輸入圖片。
- `outputs/`: 存放由擴散模型生成的圖片。
- `requirements.txt`: 專案所需的 Python 函式庫。

## 環境設定

1.  **建議使用虛擬環境：**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
2.  **安裝依賴：**
    ```bash
    pip install -r requirements.txt
    ```

## 如何執行

使用 `main.py` 腳本來執行不同的演示。

**執行 CLIP 零樣本分類演示：**

```bash
python src/main.py --demo clip --image_path images/your_test_image.jpg --labels "a photo of a cat" "a photo of a dog" "a landscape"