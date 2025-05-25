print("DEBUG: main.py 腳本已被 Python 解譯器讀取並開始執行第一行")  # 這一行我們知道可以執行了

import argparse
import torch
from clip_utils import load_clip_resources, perform_zero_shot_classification, DEFAULT_CLIP_MODEL  # 合併 import
from diffusion_utils import load_diffusion_pipeline, generate_image_from_prompt, DEFAULT_DIFFUSION_MODEL  # 合併 import


def main():
    print("DEBUG: main() 函式開始執行")
    parser = argparse.ArgumentParser(description="AI 圖像生成與分析工具")
    parser.add_argument("--demo", type=str, required=True, choices=["clip", "diffusion"],
                        help="選擇要執行的演示：'clip' 或 'diffusion'")

    # CLIP 相關參數
    parser.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_MODEL,
                        help="要使用的 CLIP 模型名稱 (來自 Hugging Face)")
    parser.add_argument("--image_path", type=str, help="[CLIP演示用] 輸入圖片的路徑")
    parser.add_argument("--labels", nargs="+", help="[CLIP演示用] 用於零樣本分類的候選文字標籤 (以空格分隔)")

    # Diffusion Model 相關參數
    parser.add_argument("--diffusion_model", type=str, default=DEFAULT_DIFFUSION_MODEL,
                        help="要使用的擴散模型 ID (來自 Hugging Face)")
    parser.add_argument("--prompt", type=str, help="[擴散模型演示用] 文字提示")
    parser.add_argument("--output_path", type=str, help="[擴散模型演示用] 生成圖片的儲存路徑")
    parser.add_argument("--num_steps", type=int, default=50, help="[擴散模型演示用] 推理步數")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="[擴散模型演示用] 引導強度")
    parser.add_argument("--negative_prompt", type=str, default=None, help="[擴散模型演示用] 負面提示")

    args = parser.parse_args()
    print(f"DEBUG: 解析到的參數: {args}")  # <--- 我們需要看到這一行

    if args.demo == "clip":
        print("DEBUG: 進入 CLIP 演示模式")  # <--- 以及這一行 (如果 demo 是 clip)
        if not args.image_path or not args.labels:
            parser.error("--image_path 和 --labels 是 CLIP 演示的必需參數。")
        print("--- 開始 CLIP 零樣本分類演示 ---")
        clip_model, clip_processor = load_clip_resources(args.clip_model)
        if clip_model and clip_processor:
            print("DEBUG: CLIP 模型和處理器已載入，準備執行分類")
            perform_zero_shot_classification(clip_model, clip_processor, args.image_path, args.labels)
        else:  # 新增一個 else 條件來捕捉模型載入失敗
            print("DEBUG: CLIP 模型或處理器載入失敗，無法執行分類。")
        print("--- CLIP 演示結束 ---")

    elif args.demo == "diffusion":
        print("DEBUG: 進入 diffusion 演示模式")  # <--- 或這一行 (如果 demo 是 diffusion)
        if not args.prompt or not args.output_path:
            parser.error("--prompt 和 --output_path 是擴散模型演示的必需參數。")
        print("--- 開始擴散模型文字生成圖像演示 ---")
        # --- 加入 PyTorch CUDA 檢測程式碼 ---
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            current_device_idx = torch.cuda.current_device()
            print(f"Current CUDA device index: {current_device_idx}")
            print(f"Current CUDA device name: {torch.cuda.get_device_name(current_device_idx)}")
        else:
            print("CUDA not available, PyTorch will use CPU.")
        # --- PyTorch CUDA 檢測程式碼結束 ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"程式決定的運行設備: {device}")
        if device == "cpu" and torch.cuda.is_available():
            print("警告：CUDA 可用，但程式似乎選擇了 CPU，請檢查邏輯！")
        elif device == "cpu":
            print("警告：在 CPU 上運行擴散模型會非常慢。")
        diffusion_pipeline = load_diffusion_pipeline(args.diffusion_model, device)
        if diffusion_pipeline:
            print(f"DEBUG: Diffusion pipeline 實際運行設備: {diffusion_pipeline.device}")
            generate_image_from_prompt(
                diffusion_pipeline,
                args.prompt,
                args.output_path,
                args.num_steps,
                args.guidance_scale,
                args.negative_prompt
            )
        else:  # 新增一個 else 條件
            print("DEBUG: 擴散模型 pipeline 載入失敗，無法生成圖像。")
        print("--- 擴散模型演示結束 ---")

    print("DEBUG: main() 函式執行完畢")  # <--- 以及這一行


if __name__ == "__main__":
    print("DEBUG: 腳本作為主程式執行 (__name__ == '__main__')")  # <--- 我們需要看到這一行
    main()  # <--- 確保 main() 被呼叫