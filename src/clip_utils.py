from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

def load_clip_resources(model_name: str = DEFAULT_CLIP_MODEL):
    """載入 CLIP 模型和處理器"""
    try:
        print(f"正在載入 CLIP 模型: {model_name}...")
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        print("CLIP 模型載入完成。")
        return model, processor
    except Exception as e:
        print(f"載入 CLIP 模型 ({model_name}) 時發生錯誤: {e}")
        return None, None

def perform_zero_shot_classification(model, processor, image_path: str, candidate_labels: list):
    """使用 CLIP 進行零樣本圖像分類"""
    if not model or not processor:
        print("CLIP 模型或處理器未成功載入。")
        return

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"錯誤：找不到圖片檔案 '{image_path}'。")
        return
    except Exception as e:
        print(f"載入圖片 '{image_path}' 時發生錯誤: {e}")
        return

    try:
        inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True, truncation=True)
    except Exception as e:
        print(f"使用 CLIP processor 處理輸入時發生錯誤: {e}")
        return

    print("\n正在計算圖片與文字的相似度...")
    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    print(f"\n--- CLIP 零樣本分類結果 (圖片: {image_path}) ---")
    results = {}
    for i, label in enumerate(candidate_labels):
        prob = probs[0, i].item()
        print(f"- \"{label}\": {prob:.4f}")
        results[label] = prob

    best_match_index = probs.argmax().item()
    best_label = candidate_labels[best_match_index]
    best_prob = probs[0, best_match_index].item()
    print(f"\n最匹配的描述是: \"{best_label}\" (機率: {best_prob:.4f})")
    return results