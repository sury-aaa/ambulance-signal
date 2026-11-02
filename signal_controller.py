import time

GREEN, YELLOW, RED = "GREEN", "YELLOW", "RED"


class SignalController:
    """Traffic signal state machine with ambulance priority."""

    def __init__(self, lanes, green_time=10, yellow_time=3,
                 all_red_time=1, priority_hold=4, clock=time.monotonic):
        self.lanes = list(lanes)
        self.green_time = green_time        # normal green length (seconds)
        self.yellow_time = yellow_time      # minimum yellow, never skipped
        self.all_red_time = all_red_time    # safety gap before priority green
        self.priority_hold = priority_hold  # keep green this long after ambulance leaves
        self.clock = clock

        self.state = "NORMAL_GREEN"
        self.current = 0                    # index of the lane that has green/yellow
        self.target = None                  # lane that will get priority green
        self.state_start = None
        self.last_seen = None

    def _go(self, state, now):
        self.state = state
        self.state_start = now

    def update(self, priority_lane=None, now=None):
        """Call once per frame. priority_lane = lane name with a confirmed ambulance, or None."""
        now = self.clock() if now is None else now
        if self.state_start is None:
            self.state_start = now
        elapsed = now - self.state_start
        want = self.lanes.index(priority_lane) if priority_lane in self.lanes else None

        if self.state == "NORMAL_GREEN":
            if want is not None:
                if want == self.current:            # ambulance is already on green
                    self.last_seen = now
                    self._go("PRIORITY_GREEN", now)
                else:                               # end current green safely via yellow
                    self.target = want
                    self._go("CLEARING_YELLOW", now)
            elif elapsed >= self.green_time:
                self._go("NORMAL_YELLOW", now)

        elif self.state == "NORMAL_YELLOW":
            if elapsed >= self.yellow_time:
                if want is not None:
                    self.target = want
                    self._go("ALL_RED", now)
                else:
                    self.current = (self.current + 1) % len(self.lanes)
                    self._go("NORMAL_GREEN", now)

        elif self.state == "CLEARING_YELLOW":
            if elapsed >= self.yellow_time:         # minimum yellow always served
                self._go("ALL_RED", now)

        elif self.state == "ALL_RED":
            if elapsed >= self.all_red_time:
                self.current = self.target
                self.last_seen = now
                self._go("PRIORITY_GREEN", now)

        elif self.state == "PRIORITY_GREEN":
            if want == self.current:
                self.last_seen = now
            elif want is None and now - self.last_seen >= self.priority_hold:
                self._go("PRIORITY_YELLOW", now)

        elif self.state == "PRIORITY_YELLOW":
            if want == self.current:                # ambulance came back
                self.last_seen = now
                self._go("PRIORITY_GREEN", now)
            elif elapsed >= self.yellow_time:
                self.current = (self.current + 1) % len(self.lanes)
                self._go("NORMAL_GREEN", now)

        return self.signals()

    def signals(self):
        colours = {lane: RED for lane in self.lanes}
        lane = self.lanes[self.current]
        if self.state in ("NORMAL_GREEN", "PRIORITY_GREEN"):
            colours[lane] = GREEN
        elif self.state in ("NORMAL_YELLOW", "CLEARING_YELLOW", "PRIORITY_YELLOW"):
            colours[lane] = YELLOW
        return colours                              # ALL_RED leaves everything red


if __name__ == "__main__":
    # Self-test with a fake clock: ambulance in Lane 3 from t=12s to t=19s
    ctl = SignalController(["Lane 1", "Lane 2", "Lane 3"])
    previous = None
    for t in range(0, 45):
        amb = "Lane 3" if 12 <= t < 20 else None
        sig = ctl.update(priority_lane=amb, now=t)
        if sig != previous:
            row = "  ".join(f"{k}:{v:6s}" for k, v in sig.items())
            print(f"t={t:2d}s  {ctl.state:16s} {row}" + ("   <- ambulance in Lane 3" if amb else ""))
            previous = sig