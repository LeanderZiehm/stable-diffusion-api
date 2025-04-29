from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch, io
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # 👈 CORS import

app = FastAPI()

# 👇 Add CORS middleware to allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    prompt: str
    steps: int = 2          # Use 2, 4, or 8
    guidance_scale: float = 0.0  # As recommended for distilled models

# app = FastAPI()

# Download & load the distilled UNet checkpoint for 4 steps
BASE = "stabilityai/stable-diffusion-xl-base-1.0"
REPO = "ByteDance/SDXL-Lightning"
CKPT = "sdxl_lightning_4step_unet.safetensors"  # Change to 2step or 8step as needed

# Load the base UNet configuration, then load our distilled weights
unet = UNet2DConditionModel.from_config(BASE, subfolder="unet").to("cuda", torch.float16)  # FP16 mixed precision :contentReference[oaicite:4]{index=4}
ckpt_path = hf_hub_download(repo_id=REPO, filename=CKPT)
unet.load_state_dict(load_file(ckpt_path, device="cuda"))  # Load with safetensors for speed & safety :contentReference[oaicite:5]{index=5}

# Build the pipeline around our distilled UNet
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE,
    unet=unet,
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")  # Deploy to GPU :contentReference[oaicite:6]{index=6}

# Configure the scheduler for trailing timesteps
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing"
)  # Ensures correct timestep ordering :contentReference[oaicite:7]{index=7}

@app.post("/generate")
def generate(payload: Payload):
    # Generate the image with distilled UNet
    generator = torch.Generator(device="cuda").manual_seed(42)
    result = pipe(
        payload.prompt,
        num_inference_steps=payload.steps,
        guidance_scale=payload.guidance_scale,
        generator=generator
    )
    img = result.images[0]

    # Stream back as PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
