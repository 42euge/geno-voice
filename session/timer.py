"""
Session timer — optional, non-intrusive time awareness.

Tracks session duration and provides gentle time-awareness signals
to prevent doom-spiraling. Does not interrupt — feeds into the
turn-taking engine as a soft signal.

The timer never forces a session to end. It offers awareness:
- After a configurable duration, marks the session as "extended"
- The turn-taking engine can use this to offer a gentle check-in
- The UI can show a subtle, non-intrusive time indicator
"""

import time
from dataclasses import dataclass


@dataclass
class TimerConfig:
    gentle_checkin_mins: float = 20.0
    extended_session_mins: float = 40.0
    checkin_interval_mins: float = 15.0


@dataclass
class TimerState:
    started_at: float
    elapsed_mins: float = 0.0
    is_extended: bool = False
    checkins_offered: int = 0
    last_checkin_at: float | None = None

    @property
    def elapsed_display(self) -> str:
        mins = int(self.elapsed_mins)
        if mins < 60:
            return f"{mins}m"
        return f"{mins // 60}h {mins % 60}m"


class SessionTimer:
    def __init__(self, config: TimerConfig | None = None):
        self.config = config or TimerConfig()
        self.state = TimerState(started_at=time.time())

    def tick(self) -> TimerState:
        now = time.time()
        self.state.elapsed_mins = (now - self.state.started_at) / 60.0
        self.state.is_extended = self.state.elapsed_mins >= self.config.extended_session_mins
        return self.state

    def should_checkin(self) -> bool:
        self.tick()

        if self.state.elapsed_mins < self.config.gentle_checkin_mins:
            return False

        if self.state.last_checkin_at is None:
            return True

        since_last = (time.time() - self.state.last_checkin_at) / 60.0
        return since_last >= self.config.checkin_interval_mins

    def record_checkin(self):
        self.state.checkins_offered += 1
        self.state.last_checkin_at = time.time()

    def checkin_message(self) -> str:
        elapsed = self.state.elapsed_display
        if self.state.checkins_offered == 0:
            return f"You've been reflecting for about {elapsed}. Take your time — just letting you know."
        if self.state.is_extended:
            return f"This session has been going for {elapsed}. It might be a good moment to pause if you'd like."
        return f"About {elapsed} now. No rush."
