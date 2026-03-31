import random
from typing import Tuple, List
from server.models import Ticket, CustomerTier, ActionType

def generate_task_1() -> Tuple[List[Ticket], List[ActionType]]:
    """Easy: Basic Keyword Routing"""
    data = [
        (Ticket(ticket_id="T01", message="I lost my password", tier=CustomerTier.STANDARD), ActionType.ROUTE_TECH),
        (Ticket(ticket_id="T02", message="Where is my invoice for last month?", tier=CustomerTier.STANDARD), ActionType.ROUTE_BILLING),
        (Ticket(ticket_id="T03", message="The app keeps crashing on startup", tier=CustomerTier.STANDARD), ActionType.ROUTE_TECH),
        (Ticket(ticket_id="T04", message="Update my credit card details", tier=CustomerTier.STANDARD), ActionType.ROUTE_BILLING),
        (Ticket(ticket_id="T05", message="Wifi signal is too weak", tier=CustomerTier.STANDARD), ActionType.ROUTE_TECH),
    ]
    tickets = [d[0] for d in data]
    expected = [d[1] for d in data]
    return tickets, expected

def generate_task_2() -> Tuple[List[Ticket], List[ActionType]]:
    """Medium: Conditional Tier-based actions
    Rule: If VIP asks for refund -> REFUND_USER. If STANDARD asks for refund -> ESCALATE_TO_HUMAN.
    """
    data = [
        (Ticket(ticket_id="T06", message="I didn't like this product, give me my money back.", tier=CustomerTier.VIP), ActionType.REFUND_USER),
        (Ticket(ticket_id="T07", message="I demand a refund right now.", tier=CustomerTier.STANDARD), ActionType.ESCALATE_TO_HUMAN),
        (Ticket(ticket_id="T08", message="Can you check my billing cycle?", tier=CustomerTier.VIP), ActionType.ROUTE_BILLING),
        (Ticket(ticket_id="T09", message="Refund my last charge please", tier=CustomerTier.VIP), ActionType.REFUND_USER),
        (Ticket(ticket_id="T10", message="I want to refund my purchase.", tier=CustomerTier.STANDARD), ActionType.ESCALATE_TO_HUMAN),
    ]
    tickets = [d[0] for d in data]
    expected = [d[1] for d in data]
    return tickets, expected

def generate_task_3() -> Tuple[List[Ticket], List[ActionType]]:
    """Hard: Contextual Escalations
    Rule: Aggressive or complex complaints must be escalated. Standard routing still applies.
    """
    data = [
        (Ticket(ticket_id="T11", message="Server is down! I am losing thousands of dollars! Fix this immediately or I will sue!", tier=CustomerTier.VIP), ActionType.ESCALATE_TO_HUMAN),
        (Ticket(ticket_id="T12", message="I need a copy of my tax invoice.", tier=CustomerTier.STANDARD), ActionType.ROUTE_BILLING),
        (Ticket(ticket_id="T13", message="Your service is a scam, cancel my account now!!!", tier=CustomerTier.STANDARD), ActionType.ESCALATE_TO_HUMAN),
        (Ticket(ticket_id="T14", message="I'm locked out of my account, please help.", tier=CustomerTier.STANDARD), ActionType.ROUTE_TECH),
        (Ticket(ticket_id="T15", message="I have a legal question regarding your terms of service.", tier=CustomerTier.VIP), ActionType.ESCALATE_TO_HUMAN),
    ]
    tickets = [d[0] for d in data]
    expected = [d[1] for d in data]
    return tickets, expected

def get_task_data(task_id: int) -> Tuple[List[Ticket], List[ActionType]]:
    if task_id == 1:
        return generate_task_1()
    elif task_id == 2:
        return generate_task_2()
    elif task_id == 3:
        return generate_task_3()
    return generate_task_1()  # fallback
