import torch
from diffusers import StableDiffusion3Pipeline
from transformers import BitsAndBytesConfig
from huggingface_hub import login
import gc

# Hugging Face 로그인
login("")  # 실제 토큰으로 변경

# 1️⃣ 4-bit 양자화 (VRAM 절약)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 8-bit보다 더 강력한 절약
    bnb_4bit_compute_dtype=torch.float16,  # 연산은 float16으로 유지
    device_map="auto"  # GPU 자동 할당
)

# 2️⃣ 모델 로드 (low_cpu_mem_usage=True로 CPU 최적화)
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.float16,  # VRAM 절약
    low_cpu_mem_usage=True,  # CPU 메모리 사용 최적화
    quantization_config=bnb_config,  # 4-bit 양자화 적용
    cache_dir="D:/huggingface"  # 캐시 저장 위치
)

# 3️⃣ VRAM 절약: 모델을 GPU + CPU 혼합 배치
pipe.enable_sequential_cpu_offload()  # 가장 강력한 VRAM 절약 옵션

# 4️⃣ Memory Efficient Attention 설정 (xFormers 대신 SDPA 사용)
#pipe.unet.set_attn_processor("sdpa")  # SDPA로 메모리 절약
pipe.enable_xformers_memory_efficient_attention()

# 5️⃣ 해상도를 낮춰서 생성 (512x512 → 이후 업스케일링)
prompt = "Spiderman fight with Scorpion in Marvel Comics"
image = pipe(prompt, height=512, width=512, num_inference_steps=20, guidance_scale=3.5).images[0]

# 6️⃣ 이미지 저장
image.save("SpiderMan.png")
print("🎉 이미지 생성 완료: capybara.png")

# 모델 삭제
del pipe

# 가비지 컬렉션 실행 (CPU 메모리 해제)
gc.collect()

# CUDA 캐시 정리 (GPU 메모리 해제)
torch.cuda.empty_cache()
torch.cuda.ipc_collect()