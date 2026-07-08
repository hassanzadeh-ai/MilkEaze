"""Training: dataset, losses, loops, and validation.

Note the asymmetry that defines the current project state:
  - the VOLUME head trains on any session with a scale (real or synthetic);
  - the CLASSIFIER trains only where per-event labels exist -> synthetic today.
"""
