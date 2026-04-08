from __future__ import annotations

import sys
import time


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TerminalTaskBar:
    def __init__(self, total: int, width: int = 28, min_interval_s: float = 0.25) -> None:
        self.total = max(1, int(total))
        self.width = max(10, int(width))
        self.min_interval_s = float(min_interval_s)
        self.start_time = time.monotonic()
        self.last_render_time = 0.0
        self.last_line_length = 0
        self.enabled = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def update(self, current: int, *, prefix: str = "", suffix: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and current < self.total and (now - self.last_render_time) < self.min_interval_s:
            return

        current = max(0, min(int(current), self.total))
        ratio = current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start_time
        remaining = (elapsed / max(ratio, 1e-8) - elapsed) if current > 0 else 0.0
        line = (
            f"{prefix} [{bar}] {current:>4d}/{self.total:<4d} "
            f"{ratio * 100:6.2f}% elapsed {format_duration(elapsed)} "
            f"remaining {format_duration(remaining)}"
        )
        if suffix:
            line += f" {suffix}"

        padded_line = line
        if len(line) < self.last_line_length:
            padded_line += " " * (self.last_line_length - len(line))

        sys.stdout.write("\r" + padded_line)
        sys.stdout.flush()
        self.last_line_length = len(line)
        self.last_render_time = now

        if current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.last_line_length = 0

    def clear(self) -> None:
        if not self.enabled or self.last_line_length <= 0:
            return
        sys.stdout.write("\r" + (" " * self.last_line_length) + "\r")
        sys.stdout.flush()
        self.last_line_length = 0

