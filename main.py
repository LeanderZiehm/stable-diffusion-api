from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch, io
from fastapi.responses import StreamingResponse

class Payload(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

app = FastAPI()
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

@app.post("/generate")
async def generate(payload: Payload):
    img = pipe(payload.prompt, height=payload.height, width=payload.width).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
