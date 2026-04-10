import os
import json
import requests
from openai import OpenAI
from server.models import TicketAction, ActionType

# ── Environment Variables ───────────────────────────────────────────────────
# The validator injects API_BASE_URL and API_KEY.
# For local testing, please ensure you export these in your terminal first.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME       = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
ENV_URL          = os.environ.get("ENV_URL", "http://127.0.0.1:8000")
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

import time

def run_task(client, task_id: int):
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")

    connected = False
    last_err = None
    for attempt in range(5):
        try:
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=10)
            res.raise_for_status()
            connected = True
            break
        except Exception as e:
            last_err = e
            time.sleep(1)
            
    if not connected:
        # We must fail loud here so the autograder knows the env container is dead,
        # instead of silently giving 0 API calls.
        raise RuntimeError(f"Failed to connect to ENV_URL {ENV_URL} after 5 retries. Error: {last_err}")

    data = res.json()
    obs = data["observation"]
    print(f"[START] task={task_name}", flush=True)
    reward = 0.0
    step_count = 0

    # Build reliable retry models list
    candidate_models = []
    if os.environ.get("MODEL_NAME"):
        candidate_models.append(os.environ["MODEL_NAME"])
    candidate_models.extend([
        "gpt-3.5-turbo", 
        "meta-llama/Llama-2-7b-chat-hf", 
        "gemini-1.5-flash", 
        "gemini-2.5-flash",
        "llama-3-8b"
    ])

    for step in range(15):
        prompt = f"Observation:\n{json.dumps(obs, indent=2)}\n\nWhat is your next action JSON?"
        
        response_text = None
        last_e = None
        if client is not None:
            for attempt_model in candidate_models:
                try:
                    completion = client.chat.completions.create(
                        model=attempt_model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0
                    )
                    response_text = completion.choices[0].message.content.strip()
                    break # Success through official proxy
                except Exception as e:
                    last_e = e
                    continue

        if not response_text:
            raise RuntimeError(f"Proxy rejected ALL models or failed to connect entirely. Last err: {last_e}")

        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()

        action = parse_model_action(response_text)
        try:
            res = requests.post(f"{ENV_URL}/step", json=action.model_dump(mode='json'))
            res.raise_for_status()
            data = res.json()
        except:
            break
        
        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]
        step_count = step + 1
        print(f"[STEP] step={step_count} reward={reward}", flush=True)

        if done:
            break

    # ── [END] block ──────────────────────────────────────────────────────────
    print(f"[END] task={task_name} score={reward} steps={step_count}", flush=True)
    return reward

def main():
    import re

    # Use re.sub to strip ALL non-printable ASCII characters (not just whitespace).
    # The validator injects URLs with \n, \r, \x00 etc. that crash httpx.InvalidURL.
    def clean(s: str) -> str:
        return re.sub(r"[^\x20-\x7E]", "", s).strip()

    api_base = clean(os.environ.get("API_BASE_URL", ""))
    api_key  = clean(os.environ.get("API_KEY", "dummy")) or "dummy"

    # Strip any trailing /chat/completions so the OpenAI SDK doesn't double-append it
    api_base = api_base.replace("/chat/completions", "")
    if api_base and not api_base.startswith("http"):
        api_base = "http://" + api_base

    print(f"Using API_BASE_URL={api_base!r}", flush=True)

    # Wrap in try/except so a malformed URL never kills the whole process.
    # The validator expects a non-zero exit only for logic errors, not bad env vars.
    try:
        client = OpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])
    except Exception as e:
        print(f"OpenAI init failed ({e}), retrying with cleaned URL…", flush=True)
        try:
            client = OpenAI(base_url=api_base, api_key=api_key)
        except Exception as e2:
            print(f"FATAL: Cannot init OpenAI client: {e2}", flush=True)
            raise

    total_score = 0.0
    for task_id in [1, 2, 3]:
        score = run_task(client, task_id)
        total_score += score

    print(f"\nTotal Score: {total_score} / 3.0", flush=True)

if __name__ == "__main__":
    main()
