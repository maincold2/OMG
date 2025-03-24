# Optimized Minimal 3D Gaussian Splatting
### Joo Chan Lee, Jong Hwan Ko, and Eunbyung Park

### [[Project Page](https://maincold2.github.io/omg/)] [[Paper(arxiv)](https://arxiv.org/abs/2503.16924)]

## Method Overview
<img src="https://github.com/maincold2/maincold2.github.io/blob/master/omg/images/fig_demo.jpg?raw=true" />

We reduce storage requirements of 3D Gaussian Splatting while using a minimal number of primitives. First, we determine the distinct Gaussian from the near ones, minimizing redundancy without sacrificing quality. Second, we propose a compact and precise attribute representation that efficiently captures both continuity and irregularity among primitives. Additionally, we propose a sub-vector quantization technique for improved irregularity representation, maintaining fast training with a negligible codebook size.

## Setup
Our code is based on [Mini-Splatting](https://github.com/fatPeter/mini-splatting).

For installation:
We recommend to use cuda 12.1 with python 3.11 for easy setup.
```shell
git clone https://github.com/maincold2/OMG.git
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
If you have trouble in installing cuml, please refer to the [CUML Installation Guide](https://docs.rapids.ai/install/).

We used [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [Tanks & Temples, and Deep Blending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

## Running

### Training

```shell
#for outdoor scenes (e.g., Mip-NeRF 360 outdoor and T&T scenes)
python train.py -s <path to COLMAP> -m <model path> --eval --imp_metric outdoor
#for indoor scenes (e.g., Mip-NeRF 360 indoor and DB scenes)
python train.py -s <path to COLMAP> -m <model path> --eval --imp_metric indoor
```
#### --importance_thresh
Threshold for importance-based scoring.
0.96, 0.98, 0.99, 0.999, 0.9999 for XS to XL. 0.96 (XS) by default

Both the ply file (3DGS format) and compressed file (comp.xz) are generated after traininig.

## Evaluation
```shell
python render.py -s <path to COLMAP> -m <model path> --decode
python metrics.py -m <model path> 
```
#### --decode
Rendering with the compressed file (comp.xz), otherwise using the ply file. The results are the same regardless of this option.

## BibTeX
```
@article{lee2025omg,
  author    = {Lee, Joo Chan and Ko, Jong Hwan and Park, Eunbyung},
  title     = {Optimized Minimal 3D Gaussian Splatting},
  journal   = {arXiv preprint arXiv:},
  year      = {2025},
}
```
