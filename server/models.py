from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Any

class ActionType(str, Enum):
    ROUTE_BILLING = "ROUTE_BILLING"
    ROUTE_TECH = "ROUTE_TECH"
    REFUND_USER = "REFUND_USER"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"

class TicketAction(BaseModel):
    action_type: ActionType

class CustomerTier(str, Enum):
    STANDARD = "Standard"
    VIP = "VIP"

class Ticket(BaseModel):
    ticket_id: str
    message: str
    tier: CustomerTier

class TicketObservation(BaseModel):
    current_ticket: Optional[Ticket]
    tickets_remaining: int
    last_feedback: str
    reward_signal: float

class TicketState(BaseModel):
    task_id: int
    queue: List[Ticket]
    expected_actions: List[ActionType]
    score: float
    turn: int
    max_turns: int
