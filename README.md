# Tensorboard Video RSL-rl logger

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-BSD%202--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

All that only because [pytorch/157712](https://github.com/pytorch/pytorch/pull/157712) is forgotten and won't be merged any time soon...

## Setup Logger

```python
logger={
  "class_name": "tensorboard_video_log_writer.TensorboardVideoLogWriter",
},
```
