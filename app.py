from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch, io
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # 👈 CORS import

class Payload(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

app = FastAPI()

# 👇 Add CORS middleware to allow all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

@app.get("/")
async def read_root():
    return {"message": "Hello"}

@app.post("/generate")
def generate(payload: Payload):
    img = pipe(prompt=payload.prompt, height=payload.height, width=payload.width).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
