---
title: Executive Assistant v1
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# OpenEnv Executive Assistant

A real-world OpenEnv environment simulating an Executive Assistant managing an inbox and a calendar.
It tests an AI Agent's ability to read emails, cross-reference calendar events, book meetings, cancel meetings to resolve conflicts, and politely reply to users.

This environment strictly meets the standard OpenEnv specifications, grading agents from 0.0 to 1.0 across three tasks of increasing difficulty.

## Tasks
1. **Inbox Zero (Easy)**: Handle simple spam and thank you emails. (Task ID: `1`)
2. **Standard Scheduling (Medium)**: Check calendar for free slots and schedule an appointment before replying. (Task ID: `2`)
3. **Conflict Resolution (Hard)**: Re-prioritize meetings, cancel a conflicting internal sync, schedule a VIP, and reply. (Task ID: `3`)

## Action Space (ExecAgentAction)
The Agent must supply a JSON object conforming to the following Pydantic Action schema:
- `action_type`: One of [`ARCHIVE_EMAIL`, `REPLY_EMAIL`, `CHECK_CALENDAR`, `SCHEDULE_MEETING`, `CANCEL_MEETING`, `END_SHIFT`]
- `email_id` (str, optional)
- `text` (str, optional)
- `date` (str, optional)
- `time` (str, optional)
- `duration` (int, optional)
- `participants` (list, optional)
- `meeting_id` (str, optional)

## Observation Space (ExecObservation)
- `inbox`: List of active emails (`email_id`, `sender`, `subject`, `body`).
- `calendar_view`: List of meetings for the dates the agent has explicitly checked.
- `last_action_result`: Feedback from the previous tool use (e.g. "Meeting Scheduled!").
- `reward_signal`: The instantaneous progress 0.0 - 1.0 on the task.

## Running Locally

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn pydantic requests openenv-core openai
   ```
2. Start the FastAPI Environment Server:
   ```bash
   uvicorn server.app:app --reload --port 8000
   ```
3. Run the baseline evaluation script using your OpenAI key:
   ```bash
   export HF_TOKEN="your_key"
   export MODEL_NAME="gpt-4o-mini"
   python inference.py
   ```

## Deploying to Hugging Face
This repository contains a `Dockerfile` that automatically creates an OpenEnv compliant inference server for containerized Hugging Face Spaces deployment over port 8000.
