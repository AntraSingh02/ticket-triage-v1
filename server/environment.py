from typing import Tuple, Dict, Any
from server.models import TicketState, TicketObservation, TicketAction, ActionType
from server.tasks import get_task_data

class TicketEnvironment:
    def __init__(self):
        self.state: TicketState = None

    def reset(self, task_id: int) -> Tuple[TicketObservation, Dict[str, Any]]:
        tickets, expected_actions = get_task_data(task_id)
        
        self.state = TicketState(
            task_id=task_id,
            queue=tickets,
            expected_actions=expected_actions,
            score=0.0,
            turn=0,
            max_turns=len(tickets)
        )
        
        obs = self._get_observation()
        # Initialize feedback to a starter message
        obs.last_feedback = "Task started! Process the queue."
        return obs, {}

    def step(self, action: TicketAction) -> Tuple[TicketObservation, float, bool, bool, Dict[str, Any]]:
        if not self.state:
            raise ValueError("Environment not initialized. Call reset() first.")
            
        if self.state.turn >= self.state.max_turns:
            return self._get_observation(), self.state.score, True, False, {}

        # Evaluate the action against the current ticket
        expected = self.state.expected_actions[self.state.turn]
        
        if action.action_type == expected:
            self.state.score += (1.0 / self.state.max_turns)
            feedback = f"✅ Correct! Ticket routed via {action.action_type.value}."
        else:
            feedback = f"❌ Incorrect. Expected {expected.value}, Agent chose {action.action_type.value}."

        self.state.turn += 1
        
        done = (self.state.turn >= self.state.max_turns)
        
        obs = self._get_observation()
        obs.last_feedback = feedback
        
        # Round score to avoid floating point weirdness
        self.state.score = round(self.state.score, 2)
        obs.reward_signal = self.state.score
        
        return obs, self.state.score, done, False, {}

    def _get_observation(self) -> TicketObservation:
        current_idx = self.state.turn
        has_ticket = current_idx < self.state.max_turns
        
        current_ticket = self.state.queue[current_idx] if has_ticket else None
        
        return TicketObservation(
            current_ticket=current_ticket,
            tickets_remaining=self.state.max_turns - current_idx,
            last_feedback="",  # Overwritten in step()
            reward_signal=self.state.score
        )
