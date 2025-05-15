from diffusers import StableDiffusionPipeline
import torch

DEFAULT_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"

def load_diffusion_pipeline(model_id: str = DEFAULT_DIFFUSION_MODEL, device: str = "cpu"):
    """載入 Stable Diffusion Pipeline"""
    try:
        print(f"正在載入 Stable Diffusion 模型: {model_id} (目標設備: {device})...")
        if device == "cuda" and torch.cuda.is_available():
            # 嘗試使用 float16 以節省 VRAM 並加速
            pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
        else:
            # CPU 或無法使用 float16
            pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)

        pipeline = pipeline.to(device)
        print("Stable Diffusion 模型載入完成。")
        # (可選) 節省 VRAM
        # if device == "cuda":
        #     pipeline.enable_attention_slicing()
        return pipeline
    except Exception as e:
        print(f"載入 Stable Diffusion 模型 ({model_id}) 時發生錯誤: {e}")
        return None

def generate_image_from_prompt(pipeline, prompt: str, output_path: str,
                               num_inference_steps: int = 50, guidance_scale: float = 7.5,
                               negative_prompt: str = None):
    """使用擴散模型根據提示生成圖片並儲存"""
    if not pipeline:
        print("擴散模型 Pipeline 未成功載入。")
        return

    print(f"\n正在根據提示生成圖像: \"{prompt}\"")
    print("這個過程可能需要一些時間，請耐心等候...")

    with torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad():
        try:
            gen_kwargs = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale
            }
            if negative_prompt:
                gen_kwargs["negative_prompt"] = negative_prompt

            output = pipeline(**gen_kwargs)
            image = output.images[0]
        except Exception as e:
            print(f"生成圖像時發生錯誤: {e}")
            print("如果是 OOM 錯誤，請嘗試在 VRAM 更大的 GPU 上運行，或啟用 attention slicing / sequential offload (如果適用)。")
            return

    try:
        image.save(output_path)
        print(f"\n圖像已成功儲存至: {output_path}")
        return output_path
    except Exception as e:
        print(f"儲存圖片 '{output_path}' 時發生錯誤: {e}")
        return None