import os
import sys

cwd = os.getcwd()
if cwd not in sys.path: sys.path.append(cwd)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from server.environment import TicketEnvironment
from server.models import TicketAction

app = FastAPI(title="Customer Support Triage Server")
env = TicketEnvironment()

class ResetRequest(BaseModel):
    task_id: int = 1

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest(task_id=1)):
    obs, info = env.reset(req.task_id)
    return {"observation": obs.model_dump(mode='json'), "info": info}

@app.post("/step")
def step(action: TicketAction):
    obs, reward, done, truncated, info = env.step(action)
    return {
        "observation": obs.model_dump(mode='json') if hasattr(obs, 'model_dump') else obs,
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info
    }

@app.get("/state")
def state():
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return {"state": env.state.model_dump(mode='json')}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/info")
def info():
    return {
        "name": "Customer Support Triage OpenEnv",
        "version": "1.0.0",
        "tasks": ["1: Basic Routing (Easy)", "2: VIP Exceptions (Medium)", "3: Anger Escalations (Hard)"]
    }

@app.get("/")
def root():
    import os
    # Dynamically find the root directory relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": f"UI file missing at {index_path}"}
