# Hierarchical Angular Separation

Code for **[Provable Cross-Modal Transfer of Hierarchical Concept Structure in CLIP Embeddings](https://openreview.net/forum?id=IFmLNK0lpW)** (NeurIPS 2026).

Hierarchical Angular Separation (HAS) means that a concept embedding is closer in angle to its descendants than to concepts on unrelated branches. Run the experiments below to generate the paper's empirical figures and table from ImageNet and synthetic data.

| Experiment | Comparison | Output directory |
| --- | --- | --- |
| Image and text hierarchy | Random, OpenAI, and DataComp CLIP embeddings | `results/figure2/` |
| Image hierarchy versus shuffled controls | Image-side HAS violations for each representation and its matched control | `results/table1/` |
| Synthetic InfoNCE transfer | HAS transfer during optimization and across regularization strengths | `results/synthetic/plots/` |

## Installation

Use Python 3.12 and run these commands from the repository root:

```bash
git clone https://github.com/nghiahhnguyen/hierarchical-angular-separation.git
cd hierarchical-angular-separation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch installation on Linux for GPU runs. To check the installation:

```bash
python -m pytest -q
```

## Prepare ImageNet

The real-data experiments use the **ImageNet-1K training split**. Obtain `ILSVRC2012_img_train.tar` from the [official ImageNet download page](https://image-net.org/download.php), following its access and usage terms.

Extract the outer archive and its class archives:

```bash
mkdir -p /path/to/imagenet/train
tar -xf /path/to/ILSVRC2012_img_train.tar -C /path/to/imagenet/train
for archive in /path/to/imagenet/train/*.tar; do
    class_dir="${archive%.tar}"
    mkdir -p "$class_dir"
    tar -xf "$archive" -C "$class_dir"
done
export IMAGENET_ROOT=/path/to/imagenet
```

The expected layout is `imagenet/train/<synset>/*.JPEG`, for example `imagenet/train/n01440764/n01440764_10026.JPEG`.

The required [hierarchy metadata](data/imagenet_hierarchy.json), class names, and prompt templates are included. No separate WordNet download is needed. Pretrained weights download automatically through [OpenAI CLIP](https://github.com/openai/CLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip) on first use. Model identifiers and measurement details are in the [experiment notes](docs/experiments.md).

## HAS measurements on ImageNet

Run the complete real-data workflow inside your compute allocation:

```bash
IMAGENET_ROOT=/path/to/imagenet DEVICE=cuda bash scripts/run_real_data.sh
```

The script extracts features from all training images, then plots image and text hierarchy and tabulates the shuffled-control comparisons. Features are cached locally under `results/cache/` for reuse. You can also extract one model at a time:

```bash
python -m has.real_data extract \
    --imagenet-root "$IMAGENET_ROOT" \
    --model openai --device cuda --batch-size 32
```

After the full extraction script has created the required caches, rerun either measurement separately:

```bash
python -m has.real_data figure2
python -m has.real_data table1
```

The image and text hierarchy plots are written as `results/figure2/figure2.png` and `figure2.pdf`. The shuffled-control comparisons are written as `results/table1/table1.md` and `table1.tex`. Both experiments also write their numeric measurements as JSON.

## HAS transfer under InfoNCE

The synthetic data are generated locally; no external download is needed. Start with the small training check:

```bash
bash scripts/smoke_synthetic.sh
```

Run the complete synthetic experiment inside your compute allocation:

```bash
DEVICE=cuda bash scripts/run_figure5.sh
```

The script generates data, optimizes the text embeddings, and plots the convergence trace and regularization sweep. It writes measurements, checkpoints, and plots under `results/synthetic/`. Use a fresh directory for each rerun, for example `RUN_ROOT=results/synthetic-rerun`.

The panel-specific settings are listed in the [experiment notes](docs/experiments.md). For individual options, use `python synthetic/run_experiment.py single --help` or `sweep --help`.
