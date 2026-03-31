import os
import json
import requests
from openai import OpenAI
from server.models import TicketAction, ActionType

API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")

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
        print(f"Failed to parse model response. Error: {e}")
        return TicketAction(action_type=ActionType.ROUTE_TECH)

def run_task(client: OpenAI, task_id: int):
    print(f"\n--- Starting Task {task_id} ---")
    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        res.raise_for_status()
    except Exception as e:
        print(f"Failed to connect to env: {e}")
        return 0.0
    
    data = res.json()
    obs = data["observation"]
    
    for step in range(15):
        prompt = f"Observation:\n{json.dumps(obs, indent=2)}\n\nWhat is your next action JSON?"
        
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            response_text = completion.choices[0].message.content.strip()
            if response_text.startswith("```json"): response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"): response_text = response_text[3:-3].strip()
        except Exception as e:
            print(f"LLM API Error: {e}")
            response_text = '{"action_type": "ROUTE_TECH"}'

        action = parse_model_action(response_text)
        print(f"[{step}] Agent Action: {action.action_type.value}")
        
        res = requests.post(f"{ENV_URL}/step", json=action.model_dump(mode='json'))
        data = res.json()
        obs = data["observation"]
        reward = data["reward"]
        done = data["done"]
        print(f"    Feedback: {obs.get('last_feedback')}")
        
        if done:
            print(f"Task {task_id} Finished. Final Score: {reward}")
            return reward

    return reward

def main():
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except: return

    total_score = 0.0
    for task_id in [1, 2, 3]:
        score = run_task(client, task_id)
        total_score += score
        
    print(f"\nTotal Score: {total_score} / 3.0")

if __name__ == "__main__":
    main()
