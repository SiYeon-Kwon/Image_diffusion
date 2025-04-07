import torch
import gc
from diffusers import StableDiffusion3Pipeline, StableVideoDiffusionPipeline
from transformers import BitsAndBytesConfig
from huggingface_hub import login

# 🔹 Hugging Face 로그인 (토큰 입력 필요)
login("")

# 🔹 1️⃣ 4-bit 양자화 설정 (VRAM 절약)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit 양자화 적용
    bnb_4bit_compute_dtype=torch.float16,  # float16 연산 유지
    device_map="auto"  # GPU 자동 할당
)

# 🔹 2️⃣ Stable Diffusion 3.5 모델 로드
pipe_sd = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    quantization_config=bnb_config,
    cache_dir="D:/huggingface"
)
pipe_sd.enable_sequential_cpu_offload()  # VRAM 절약

# 🔹 3️⃣ 이미지 생성 (512x512 해상도)
prompt = "Spiderman fight with Captain of America"
image = pipe_sd(prompt, height=512, width=512, num_inference_steps=20, guidance_scale=3.5).images[0]

# 🔹 4️⃣ 이미지 저장
image_path = "SpiderMan.png"
image.save(image_path)
print(f"🎉 이미지 생성 완료: {image_path}")

# 🔹 5️⃣ Stable Video Diffusion (SVD) 모델 로드
pipe_svd = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16
).to("cuda")

# 🔹 6️⃣ 이미지 기반 동영상 생성
video_frames = pipe_svd(image_path, decode_chunk_size=4).frames

# 🔹 7️⃣ 동영상 저장 (GIF 또는 MP4)
video_frames[0].save("spiderman_fight.gif", save_all=True, append_images=video_frames[1:], duration=100, loop=0)
print("🎥 동영상 생성 완료: spiderman_fight.gif")

# 🔹 메모리 정리
del pipe_sd, pipe_svd
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
