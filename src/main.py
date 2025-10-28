import argparse
import torch
from clip_utils import load_clip_resources, perform_zero_shot_classification, DEFAULT_CLIP_MODEL  # 合併 import
from diffusion_utils import load_diffusion_pipeline, generate_image_from_prompt, DEFAULT_DIFFUSION_MODEL  # 合併 import


def main():
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

    if args.demo == "clip":
        if not args.image_path or not args.labels:
            parser.error("--image_path 和 --labels 是 CLIP 演示的必需參數。")
        
        print("--- 開始 CLIP 零樣本分類演示 ---")
        clip_model, clip_processor = load_clip_resources(args.clip_model)
        
        if clip_model and clip_processor:
            perform_zero_shot_classification(clip_model, clip_processor, args.image_path, args.labels)
        else:
            print("錯誤：CLIP 模型或處理器載入失敗，無法執行分類。")
        print("--- CLIP 演示結束 ---")

    elif args.demo == "diffusion":
        if not args.prompt or not args.output_path:
            parser.error("--prompt 和 --output_path 是擴散模型演示的必需參數。")
        
        print("--- 開始擴散模型文字生成圖像演示 ---")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cpu":
            print("警告：未檢測到 CUDA。在 CPU 上運行擴散模型會非常慢。")
            
        diffusion_pipeline = load_diffusion_pipeline(args.diffusion_model, device)
        
        if diffusion_pipeline:
            generate_image_from_prompt(
                diffusion_pipeline,
                args.prompt,
                args.output_path,
                args.num_steps,
                args.guidance_scale,
                args.negative_prompt
            )
        else:
            print("錯誤：擴散模型 pipeline 載入失敗，無法生成圖像。")
        print("--- 擴散模型演示結束 ---")


if __name__ == "__main__":
    main()