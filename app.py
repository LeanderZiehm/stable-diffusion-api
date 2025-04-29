from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch, io
from fastapi.responses import StreamingResponse

class Payload(BaseModel):
    prompt: str
    steps: int = 4
    guidance_scale: float = 0.0

app = FastAPI()

# ✅ Enable CORS (Allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 Model setup
BASE = "stabilityai/stable-diffusion-xl-base-1.0"
REPO = "ByteDance/SDXL-Lightning"
CKPT = "sdxl_lightning_4step_unet.safetensors"

# Load full SDXL pipeline first (ensures all configs are fetched)
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE,
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

# Load distilled UNet weights into existing pipeline
ckpt_path = hf_hub_download(repo_id=REPO, filename=CKPT)
state_dict = load_file(ckpt_path, device="cuda")
pipe.unet.load_state_dict(state_dict, strict=False)

# Set trailing timestep Euler scheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing"
)

@app.get("/")
async def read_root():
    return {"message": "Hello from SDXL-Lightning"}

@app.post("/generate")
def generate(payload: Payload):
    generator = torch.Generator(device="cuda").manual_seed(42)
    result = pipe(
        payload.prompt,
        num_inference_steps=payload.steps,
        guidance_scale=payload.guidance_scale,
        generator=generator
    )
    img = result.images[0]

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
