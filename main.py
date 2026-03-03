
# render_backend/main.py
# Deploy this on Render (Free Web Service)
# Requirements: fastapi uvicorn diffusers torch numpy pillow

import io, json, base64, time
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from diffusers import DDPMPipeline, DDIMScheduler

app = FastAPI(title="EDBA Inference API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ──
DEVICE = "cpu"  # Render free tier = CPU only
pipeline = None
scheduler = None

@app.on_event("startup")
async def load_model():
    global pipeline, scheduler
    print("[Startup] Loading model...")
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(DEVICE)
    print("[Startup] Model loaded.")

# ── Request / Response schemas ──
class GenerateRequest(BaseModel):
    budget: int = 50
    pilot_steps: int = 5
    alpha: float = 0.6
    patch_size: int = 8
    t_min: int = 3
    t_max: int = 40
    seed: int = 42
    use_edba: bool = True      # False = DDIM baseline for comparison

class GenerateResponse(BaseModel):
    image_b64: str
    allocation_map: list
    entropy_map: list
    fid_estimate: float
    time_seconds: float
    edba_used: bool
    budget_used: int

# ── Entropy computation ──
def compute_entropy(feat, grid_h, grid_w):
    if feat.shape[-1] != grid_w or feat.shape[-2] != grid_h:
        feat = F.interpolate(feat, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
    B, C, H, W = feat.shape
    flat = feat.reshape(B, C, -1)
    probs = F.softmax(flat, dim=1).clamp(1e-8)
    entropy = -(probs * probs.log()).sum(dim=1).reshape(B, H, W)
    e = entropy.cpu().numpy()
    for b in range(B):
        mn, mx = e[b].min(), e[b].max()
        if mx > mn:
            e[b] = (e[b] - mn) / (mx - mn)
    return e

# ── Fractional Knapsack allocation ──
def allocate(urgencies, budget, t_min, t_max):
    m = len(urgencies)
    alloc = np.full(m, float(t_min))
    remaining = float(budget - t_min * m)
    if remaining <= 0:
        return alloc
    total_u = urgencies.sum() + 1e-8
    sorted_idx = np.argsort(-urgencies)
    for idx in sorted_idx:
        extra = (urgencies[idx] / total_u) * remaining
        alloc[idx] += extra
    alloc = np.clip(alloc, t_min, t_max)
    alloc *= budget / alloc.sum()
    return np.clip(alloc, t_min, t_max)

# ── Laplacian smoothing ──
def smooth(alloc_flat, grid_h, grid_w, lam=0.3, iters=3):
    A = alloc_flat.copy().reshape(grid_h, grid_w)
    for _ in range(iters):
        A_new = A.copy()
        for r in range(grid_h):
            for c in range(grid_w):
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        neighbors.append(A[nr, nc])
                if neighbors:
                    A_new[r,c] = A[r,c] + lam * (np.mean(neighbors) - A[r,c])
        A = A_new
    return A.reshape(-1)

# ── Main generation endpoint ──
@torch.no_grad()
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    try:
        torch.manual_seed(req.seed)
        scheduler.set_timesteps(req.budget)
        timesteps = scheduler.timesteps
        grid = 64 // req.patch_size

        unet = pipeline.unet
        x = torch.randn(1, unet.config.in_channels,
                        unet.config.sample_size, unet.config.sample_size)

        activations = {}
        def hook(m, i, o):
            activations["mid"] = (o[0] if isinstance(o, tuple) else o).detach()
        h = unet.mid_block.register_forward_hook(hook)

        t_start = time.time()
        alloc_map = np.ones(grid*grid) * (req.budget - req.pilot_steps) / (grid*grid)
        entropy_out = np.ones((grid, grid))

        if req.use_edba:
            # Phase 1: Pilot
            for t in timesteps[:req.pilot_steps]:
                tb = torch.tensor([t])
                np_ = unet(x, tb).sample
                x = scheduler.step(np_, t, x).prev_sample

            entropy_map = compute_entropy(activations.get("mid", torch.ones(1,64,grid,grid)), grid, grid)
            entropy_out = entropy_map[0]

            # Phase 2: Allocate
            urgency = np.clip(entropy_out.reshape(-1), 0, 1)
            alloc = allocate(urgency, req.budget - req.pilot_steps, req.t_min, req.t_max)

            # Phase 3: Smooth
            alloc_smooth = smooth(alloc, grid, grid, lam=0.3)
            alloc_map = alloc_smooth
            alloc_grid = alloc_smooth.reshape(grid, grid)

            # Phase 4: Non-uniform denoising
            steps_used = np.zeros((grid, grid))
            for t in timesteps[req.pilot_steps:]:
                tb = torch.tensor([t])
                np_ = unet(x, tb).sample
                mask = np.zeros((grid, grid))
                for r in range(grid):
                    for c in range(grid):
                        if steps_used[r,c] < alloc_grid[r,c]:
                            mask[r,c] = 1.0
                            steps_used[r,c] += 1
                mask_t = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
                mask_t = F.interpolate(mask_t, size=(unet.config.sample_size, unet.config.sample_size), mode='nearest')
                x_new = scheduler.step(np_, t, x).prev_sample
                x = x * (1 - mask_t) + x_new * mask_t
        else:
            # Plain DDIM
            for t in timesteps:
                tb = torch.tensor([t])
                np_ = unet(x, tb).sample
                x = scheduler.step(np_, t, x).prev_sample

        h.remove()
        elapsed = time.time() - t_start

        # Convert image to base64
        img_np = ((x[0].permute(1,2,0).numpy().clip(-1,1) + 1) / 2 * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return GenerateResponse(
            image_b64=img_b64,
            allocation_map=alloc_map.tolist(),
            entropy_map=entropy_out.tolist(),
            fid_estimate=0.0,  # full FID needs bulk evaluation
            time_seconds=elapsed,
            edba_used=req.use_edba,
            budget_used=req.budget
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "model": "ddpm-celebahq", "algorithm": "EDBA"}

@app.get("/algorithm_info")
async def algorithm_info():
    return {
        "name": "EDBA: Entropy-Driven Denoising Budget Allocation",
        "phases": {
            "phase1": "Pilot entropy estimation — O(pilot_steps × forward_pass)",
            "phase2": "Fractional knapsack allocation — O(m log m)",
            "phase3": "Laplacian graph smoothing — O(iterations × m)",
            "phase4": "Non-uniform denoising — O(budget × forward_pass)"
        },
        "complexity": "O(m log m) algorithmic overhead",
        "paper_claim": "First spatial budget allocation algorithm for diffusion inference"
    }

# Run: uvicorn main:app --host 0.0.0.0 --port 10000
