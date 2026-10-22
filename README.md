# SCL-ViT

Reference implementation scaffold for **Spatially Conditioned and Locally Refined Diffusion Transformers for Text-to-Thangka Generation**.

This repository implements SPE, CAE and LFR, a compact U-shaped Transformer denoiser, and training/sampling/evaluation entry points. The paper does not identify all pretrained component revisions and sampling details; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before interpreting results as an exact reproduction.

## Setup

Python 3.10+ and PyTorch 2.x are recommended. Install with `pip install -e '.[eval]'`. Install a CUDA-enabled PyTorch build appropriate for your machine separately. First run downloads the configured pretrained weights from Hugging Face.

## Data format

Create a JSONL manifest with one record per image:

```json
{"image":"images/0001.png","caption":"A central Buddha on a lotus throne..."}
```

Image paths may be absolute or relative to the manifest. Keep train/validation/test manifests separate. Do not redistribute T2IThangka or Oxford 102 assets without checking their terms.

## Train

Training uses the pretrained Stable Diffusion 1.5 VAE and CLIP ViT-L/14 text encoder by default; override the IDs in `configs/thangka.yaml` if matching another experiment setup:

```bash
python -m scl_vit.train --config configs/thangka.yaml
```

## Sample

```bash
python -m scl_vit.sample --config configs/thangka.yaml --checkpoint runs/scl_vit/best.pt \
  --prompt "A richly ornamented Tibetan thangka..." --output outputs/sample.png
```

## Evaluate

After generating the same number of images as the held-out split, run `python -m scl_vit.evaluate --real data/test_images --generated outputs/test_samples`. Use identical preprocessing and feature extraction for all baselines. This FID command does not calculate CLIPScore or LPIPS.

## Method

- **SPE** adds a learned supplementary positional embedding to the U-ViT token sequence and exposes image-token positions to CAE.
- **CAE** combines channel and spatial modulation, positional features, and a depthwise-convolution-derived gate through a residual update.
- **LFR** applies two depthwise convolutions with GELU and a residual connection to image tokens.

See `scl_vit/models/modules.py` for standalone modules and `scl_vit/models/scl_vit.py` for the backbone adapter.

## Citation

```bibtex
@article{xue2026sclvit,
  title   = {SCL-ViT: Spatially Conditioned and Locally Refined Diffusion Transformers for Text-to-Thangka Generation},
  author  = {Xue, Panpan and Hu, Wenjin and Zhao, Yan},
  year    = {2026},
  note    = {Manuscript; publication details to be added}
}
```

## License

Code is provided under the MIT License. Dataset and pretrained-model licenses remain with their respective owners.
