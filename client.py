import requests
from typing import Tuple, Dict, Any
from server.models import ExecAgentAction, ExecObservation

class ExecAssistantClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
            
    def reset(self, task_id: int = 1) -> Tuple[ExecObservation, Dict[str, Any]]:
        res = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
        res.raise_for_status()
        data = res.json()
        return ExecObservation(**data["observation"]), data["info"]

    def step(self, action: ExecAgentAction) -> Tuple[ExecObservation, float, bool, bool, Dict[str, Any]]:
        payload = action.model_dump(mode='json') if hasattr(action, 'model_dump') else action.dict()
        res = requests.post(f"{self.base_url}/step", json=payload)
        res.raise_for_status()
        data = res.json()
        return (
            ExecObservation(**data["observation"]),
            data["reward"],
            data["done"],
            data["truncated"],
            data["info"]
        )
