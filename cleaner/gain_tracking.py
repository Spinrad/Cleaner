"""Gain bookkeeping along the DSP chain.

Predicts peak/RMS level after each stage by applying stage gain offsets.
This is a linear gain ledger — it does NOT model frequency-dependent
loudness, compression/saturation non-linearities, or crest factor changes.
Use as a coarse heuristic; actual peak/RMS must be measured on the output.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GainStage:
    name: str
    gain_db: float
    description: str = ""


class GainTracker:
    """Tracks predicted peak and RMS levels through DSP stages."""
    
    def __init__(self, initial_peak_dbfs: float, initial_rms_dbfs: float):
        self._initial_peak = initial_peak_dbfs
        self._initial_rms = initial_rms_dbfs
        self.current_peak_dbfs = initial_peak_dbfs
        self.current_rms_dbfs = initial_rms_dbfs
        self.stages: list[GainStage] = []
    
    def predict_after(self, gain_db: float) -> tuple[float, float]:
        """Predict peak and RMS after applying gain_db, without committing."""
        return (
            self.current_peak_dbfs + gain_db,
            self.current_rms_dbfs + gain_db,
        )
    
    def commit(self, name: str, gain_db: float, description: str = ""):
        """Record a stage and update the current level."""
        self.stages.append(GainStage(name=name, gain_db=gain_db,
                                      description=description))
        self.current_peak_dbfs += gain_db
        self.current_rms_dbfs += gain_db
    
    @property
    def total_gain_db(self) -> float:
        return sum(s.gain_db for s in self.stages)
    
    @property
    def crest_db(self) -> float:
        return self.current_peak_dbfs - self.current_rms_dbfs
