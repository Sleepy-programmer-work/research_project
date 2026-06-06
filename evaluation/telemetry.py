import pynvml
import psutil
import threading
import time
import torch
from typing import Dict

class PeakResourceTracker:
    """
    Thread-based peak sampler for GPU VRAM, GPU utilisation, and RAM.
    Use as a context manager to guarantee correct scoping:

        with PeakResourceTracker(device_index=0) as tracker:
            run_pipeline(...)
        stats = tracker.stats   # available after __exit__
    """

    POLL_INTERVAL_S = 0.05   # 50 ms — fine enough for short GPU bursts

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.stats: Dict[str, float] = {}
        self._running = False
        self._peak_vram_mb = 0.0
        self._peak_ram_mb = 0.0
        self._peak_util_pct = 0.0
        self._baseline_ram_mb = 0.0
        self._baseline_vram_mb = 0.0

        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._nvml_ok = True
        except pynvml.NVMLError:
            self._nvml_ok = False

    def _poll(self):
        process = psutil.Process()
        while self._running:
            # --- RAM (incremental delta from baseline) ---
            rss_mb = process.memory_info().rss / (1024 * 1024)
            delta_mb = rss_mb - self._baseline_ram_mb
            if delta_mb > self._peak_ram_mb:
                self._peak_ram_mb = delta_mb

            # --- GPU VRAM and utilisation ---
            if self._nvml_ok:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    vram_mb = mem.used / (1024 * 1024)
                    if vram_mb > self._peak_vram_mb:
                        self._peak_vram_mb = vram_mb

                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    if util.gpu > self._peak_util_pct:
                        self._peak_util_pct = float(util.gpu)
                except Exception:
                    pass

            time.sleep(self.POLL_INTERVAL_S)

    def __enter__(self):
        self._peak_vram_mb = 0.0
        self._peak_ram_mb = 0.0
        self._peak_util_pct = 0.0
        # Snapshot baseline RAM *before* pipeline starts
        self._baseline_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        # Snapshot baseline VRAM *before* pipeline starts
        self._baseline_vram_mb = 0.0
        if self._nvml_ok:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._baseline_vram_mb = mem.used / (1024 * 1024)
            except Exception:
                pass
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        self._thread.join(timeout=2.0)
        # VRAM delta: peak observed minus baseline (isolates pipeline allocation)
        vram_delta = max(0.0, self._peak_vram_mb - self._baseline_vram_mb)
        self.stats = {
            "peak_vram_mb":    round(vram_delta, 2),
            "peak_ram_delta_mb": round(self._peak_ram_mb, 2),
            "peak_gpu_util_pct": round(self._peak_util_pct, 1),
        }
        return False   # do not suppress exceptions
