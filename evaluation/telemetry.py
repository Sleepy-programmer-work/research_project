import pynvml
import psutil
import threading
import time
import torch
import logging
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

            # --- GPU VRAM delta (real-time, relative to pre-pipeline baseline) ---
            # Compute the delta on every tick so _peak_vram_mb always holds the
            # highest *incremental* VRAM seen, never the absolute GPU memory.
            if self._nvml_ok:
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    vram_delta_mb = mem.used / (1024 * 1024) - self._baseline_vram_mb
                    if vram_delta_mb > self._peak_vram_mb:
                        self._peak_vram_mb = vram_delta_mb

                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    if util.gpu > self._peak_util_pct:
                        self._peak_util_pct = float(util.gpu)
                except Exception:
                    pass

            time.sleep(self.POLL_INTERVAL_S)


    def __enter__(self):
        # _peak_vram_mb starts at 0.0 — _poll() will track the per-tick delta
        # (absolute VRAM minus baseline), so 0.0 is the correct floor here.
        self._peak_vram_mb = 0.0
        self._peak_ram_mb = 0.0
        self._peak_util_pct = 0.0
        # Snapshot baseline RAM *before* pipeline starts
        self._baseline_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        # Snapshot baseline VRAM *before* pipeline starts so _poll() can
        # subtract it on every tick, yielding pure incremental delta values.
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
        # _peak_vram_mb already holds the highest observed delta (computed
        # per-tick in _poll), so assign it directly — no further subtraction.
        peak_ram_delta_mb = round(self._peak_ram_mb, 2)
        assert peak_ram_delta_mb >= 0, "Negative RAM delta — baseline captured after allocation"
        logging.getLogger(__name__).debug(f"Peak RAM delta: {peak_ram_delta_mb:.1f} MB")
        self.stats = {
            "peak_vram_mb":      round(max(0.0, self._peak_vram_mb), 2),
            "peak_ram_delta_mb": peak_ram_delta_mb,
            "peak_gpu_util_pct": round(self._peak_util_pct, 1),
        }
        return False   # do not suppress exceptions
