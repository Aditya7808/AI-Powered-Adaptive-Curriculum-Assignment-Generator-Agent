import time


class StageTimer:
    def __init__(self, timings: dict[str, float], stage_name: str):
        self.timings = timings
        self.stage_name = stage_name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.timings[self.stage_name] = elapsed_ms
