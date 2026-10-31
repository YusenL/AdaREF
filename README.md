# AdaREF

Core implementation of **AdaREF: Adaptive reference-editing fusion for cross-device wearable heart-rate estimation**

## Structure

- `model.py`: AdaREF model, source loss, and HR dynamics.
- `adaptation.py`: EDTA updates.
- `main.py`: synthetic demo.
- `configs/paper.json`: paper settings.
- `tests/`: core checks.

## Installation

Python 3.9–3.11.

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py --config configs/paper.json
python -m unittest discover -s tests -v
```

The demo uses synthetic inputs and random weights and runs two EDTA steps. Add `--steps 100` to use the paper's update budget. It checks the algorithms rather than reproducing benchmark results.

BeliefPPG attribution and license are included in [model.py](model.py).
