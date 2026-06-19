# EPPINN: Evidential Physics-Informed Neural Networks

> CT Perfusion uncertainty quantification — MICCAI 2026
>
> Paper: https://arxiv.org/abs/2603.09359

A minimal reference implementation of Evidential Perfusion PINNs for CT Perfusion
parameter estimation with voxel-wise uncertainty.

## Files

- `train.py` — training / inference entry-point
- `model.py` — EPPINN architecture (hash encoding + SIREN + evidential head)
- `data.py` — ISLES2018 case loader
- `utils.py` — seed setting and physiological clipping helpers

## Install

```bash
pip install -r requirements.txt
# tiny-cuda-nn (CUDA required, build from source):
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn
cd tiny-cuda-nn/bindings/torch && python setup.py install
```

## Data

ISLES2018 dataset: https://www.smir.ch/ISLES/Start2018

Organize each case as `<DATA_ROOT>/<case_id>/` containing NIfTI volumes
(`CTP_4D.nii.gz`, `aif.npy`, `brainmask.nii.gz`) as expected by `data.py`.

## Run

```bash
DATA_ROOT=/path/to/isles python train.py \
    --case_dir $DATA_ROOT/case_001 \
    --output_dir results
```

## Cite

```bibtex
@misc{lee2026evidentialperfusionphysicsinformedneural,
  title={Evidential Perfusion Physics-Informed Neural Networks with Residual Uncertainty Quantification},
  author={Junhyeok Lee and Minseo Choi and Han Jang and Young Hun Jeon and Heeseong Eum and Joon Jang and Chul-Ho Sohn and Kyu Sung Choi},
  year={2026},
  eprint={2603.09359},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.09359}
}
```

## License

MIT
