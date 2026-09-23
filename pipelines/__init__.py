"""
Sampling pipelines.

- `standard`: the corrected pipeline. One sigma schedule for the whole run,
  split into segments where shader noise enters, honouring denoise and custom
  sigmas, with noise kept in the distribution the model expects.
- legacy: the pre-2.0 pipeline, still living in `shader_noise_ksampler.py` and
  frozen there. Workflows saved before 2.0 keep using it so their seeds
  reproduce; CHANGELOG.md's 2.0.0 section lists what it gets wrong.
"""

from . import standard

__all__ = ["standard"]
