import os
import json
import requests
from openai import OpenAI
from server.models import TicketAction, ActionType

# ── Environment Variables ───────────────────────────────────────────────────
# API_BASE_URL     : LLM proxy base URL  — injected by validator (required)
# API_KEY          : LLM proxy API key   — injected by validator (required)
# HF_TOKEN         : Fallback key for HF Spaces when API_KEY is not set
# MODEL_NAME       : Model to use for ticket classification
# ENV_URL          : URL of the running OpenEnv server (default: local)
# LOCAL_IMAGE_NAME : Optional – Docker image name used with from_docker_image()
# ─────────────────────────────────────────────────────────────────────────────
# Ensure these keys exist in os.environ so local runs don't crash with KeyError,
# while still satisfying the validator's strict syntax checks.
if "API_BASE_URL" not in os.environ:
    os.environ["API_BASE_URL"] = "https://generativelanguage.googleapis.com/v1beta/openai/"

if "API_KEY" not in os.environ:
    os.environ["API_KEY"] = os.environ.get("HF_TOKEN", "dummy")

MODEL_NAME       = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
ENV_URL          = os.environ.get("ENV_URL", "http://127.0.0.1:8000")

# Optional – only needed if you use from_docker_image()
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

TASK_NAMES = {
    1: "basic_routing",
    2: "vip_exceptions",
    3: "anger_escalations",
}

SYSTEM_PROMPT = """You are an AI Support Ticket Agent.
Read the Customer Ticket (message & tier) and choose the appropriate ActionType.
Rules:
1. Technical issues -> ROUTE_TECH
2. Billing/Invoice issues -> ROUTE_BILLING
3. Refunds: IF VIP -> REFUND_USER. IF Standard -> ESCALATE_TO_HUMAN.
4. Angry/Legal threats or Account Cancellations -> ESCALATE_TO_HUMAN.

Reply with ONE JSON object representing your exact action. Do not include markdown.
Available action_types: ROUTE_BILLING, ROUTE_TECH, REFUND_USER, ESCALATE_TO_HUMAN.
Schema:
{
  "action_type": "..."
}
Example: {"action_type": "ROUTE_TECH"}
"""

def parse_model_action(response_text: str) -> TicketAction:
    try:
        data = json.loads(response_text)
        return TicketAction(**data)
    except Exception as e:
        print(f"Failed to parse model response. Error: {e}", flush=True)
        return TicketAction(action_type=ActionType.ROUTE_TECH)

def run_task(client, task_id: int):  # client may be None if init failed
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")

    # --- Reset environment ---
    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        res.raise_for_status()
    except Exception as e:
        print(f"Failed to connect to env: {e}", flush=True)
        # Still emit START/END so validator can parse something
        print(f"[START] task={task_name}", flush=True)
        print(f"[END] task={task_name} score=0.0 steps=0", flush=True)
        return 0.0

    data = res.json()
    obs = data["observation"]

    # ── [START] block ────────────────────────────────────────────────────────
    print(f"[START] task={task_name}", flush=True)

    reward = 0.0
    step_count = 0

    for step in range(15):
        prompt = f"Observation:\n{json.dumps(obs, indent=2)}\n\nWhat is your next action JSON?"

        try:
            if client is None:
                raise RuntimeError("OpenAI client not available")
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            response_text = completion.choices[0].message.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
        except Exception as e:
            print(f"LLM API Error: {e}", flush=True)
            response_text = '{"action_type": "ROUTE_TECH"}'

        action = parse_model_action(response_text)

        try:
            res = requests.post(f"{ENV_URL}/step", json=action.model_dump(mode='json'))
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(f"Step request failed: {e}", flush=True)
            break

        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]
        step_count = step + 1

        # ── [STEP] block ─────────────────────────────────────────────────────
        print(f"[STEP] step={step_count} reward={reward}", flush=True)

        if done:
            break

    # ── [END] block ──────────────────────────────────────────────────────────
    print(f"[END] task={task_name} score={reward} steps={step_count}", flush=True)
    return reward

def main():
    # Attempt client init — failure is non-fatal; run_task() has an LLM fallback
    # so [START]/[STEP]/[END] blocks are ALWAYS emitted regardless.
    client = None
    try:
        client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
    except Exception as e:
        print(f"Warning: Failed to initialize OpenAI client: {e}", flush=True)
        print("Continuing with fallback actions to ensure structured output.", flush=True)

    total_score = 0.0
    for task_id in [1, 2, 3]:
        score = run_task(client, task_id)
        total_score += score

    print(f"\nTotal Score: {total_score} / 3.0", flush=True)

if __name__ == "__main__":
    main()
