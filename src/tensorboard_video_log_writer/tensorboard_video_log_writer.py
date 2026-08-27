from __future__ import annotations

import pathlib
import tempfile
from typing import Optional

import torch
from rsl_rl.utils.log_writer import LogWriter
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter

try:
  from moviepy import editor as mpy  # MoviePy 1.x
except ImportError:
  try:
    import moviepy as mpy  # MoviePy ≥2.0
  except ImportError:
    mpy = None


class TensorboardVideoLogWriter(SummaryWriter, LogWriter):
  """Summary writer for Tensorboard with GIF video support."""

  def __init__(self, log_dir: str) -> None:
    super().__init__(log_dir, flush_secs=10)

    # Initialize set to keep track of logged videos
    self.logged_videos: set[str] = set()

  def save_video(self, video: pathlib.Path, it: int) -> None:
    """Upload a video artifact once per filename to Tensorboard."""
    if mpy is None:
      return
    if video.name not in self.logged_videos:
      torch._C._log_api_usage_once("tensorboard.logging.add_video")
      video_summary = make_video(video)
      self._get_file_writer().add_summary(
        Summary(value=[Summary.Value(tag="video", image=video_summary)]), it, None
      )
      self.logged_videos.add(video.name)


# Adapted from https://github.com/pytorch/pytorch/pull/157712
def _safe_write_gif(clip, path: str) -> None:
  """Try the various MoviePy write_gif signatures until one works."""
  variants = [
    dict(verbose=False, logger=None),  # MoviePy ≥1.0.2
    dict(verbose=False, progress_bar=False),  # MoviePy 1.0.0 / 1
    dict(verbose=False),  # fallback
  ]
  for kwargs in variants:
    try:
      clip.write_gif(path, **kwargs)
      return
    except TypeError:
      continue
  # last resort
  clip.write_gif(path)


def make_video(video: pathlib.Path) -> Summary.Image | None:
  """Convert a (T,H,W,C) uint8/float32 tensor to a GIF and wrap as Summary.Image."""
  # ---- MoviePy import shim -------------------------------------------------
  clip = mpy.VideoFileClip(video)
  try:
    # write/read the GIF inside one with-block
    with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
      _safe_write_gif(clip, tmp.name)
      tmp.seek(0)
      gif_bytes = tmp.read()
    return Summary.Image(
      height=clip.h,
      width=clip.w,
      colorspace=clip.get_frame(0).shape[2],
      encoded_image_string=gif_bytes,
    )
  finally:
    clip.close()
