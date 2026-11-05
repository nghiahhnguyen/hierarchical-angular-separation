# Experiment settings

## Real-data measurements

| Key | Implementation | Architecture | Weights |
| --- | --- | --- | --- |
| `random_vit` | OpenCLIP | `ViT-L-14` | Random initialization; explicit seed |
| `openai` | OpenAI CLIP | `ViT-L/14` | OpenAI pretrained checkpoint |
| `datacomp1b` | OpenCLIP | `ViT-L-14` | `datacomp_xl_s13b_b90k` |
| `raw_pixels` | CLIP image preprocessing | Resized and center-cropped RGB pixels | No model |

For each eligible anchor, HAS-Viol is the percentage of off-branch concepts whose angle from the anchor is smaller than its descendant-cone boundary. Off-branch concepts exclude the anchor, its descendants, and its ancestors. The boundary is the 95th percentile of descendant angles. Unary hierarchy nodes are collapsed before measurement.

For the image and text hierarchy comparison, text concepts use their own names with the ImageNet prompt templates. Each prompt embedding is normalized, then averaged for that concept. Image concepts use count-weighted means over their distinct descendant leaves. Box plots group per-anchor violation percentages by depth and hide outliers.

The shuffled-control comparison uses disjoint even/odd image sets within each leaf. Anchors from split A are compared with descendants and off-branch concepts from split B, and the reverse direction is evaluated separately. The two directions are averaged per anchor, then percentages are averaged across eligible anchors. Internal prototypes are count-weighted means.

The shuffled control permutes leaf embeddings together with their image counts, uses the same permutation in both splits, and preserves the hierarchy topology. Gaussian instance vectors are independently drawn, normalized, and aggregated using ImageNet's leaf counts.

| Representation | Original hierarchy | Shuffled hierarchy |
| --- | --- | --- |
| Raw pixels | One extraction | Population SD over 5 leaf shuffles |
| Random ViT | Sample SD over 3 initialization seeds | Sample SD over 3 initialization seeds × 5 leaf shuffles |
| OpenAI CLIP | One extraction | Population SD over 5 leaf shuffles |
| Gaussian vectors | Population SD over 40 independent draws | Population SD over 5 draws × 5 leaf shuffles |

Fresh extraction fixes per-class image ordering and accumulates means in float32. Historical per-image cache ordering and numerical reductions are not preserved, so fresh runs can differ from the paper's last printed digits.

Use `--text-only` for text embeddings or `--skip-text` for image-only extraction. For a smaller extraction check, use a positive `--max-images` with its own `--cache-root`; its measurements are not comparable with the full-data table.

## Synthetic InfoNCE

The fixed-image unconstrained features model learns only the text embeddings. Each image has one caption that remains fixed throughout optimization.

| Setting | Convergence panel | Regularization panel |
| --- | --- | --- |
| Tree | Depth 5, branching factor 3 | Same |
| Feature dimension | Number of hierarchy nodes | Same |
| Images per leaf | 150 | Same |
| Prototype angles | Geometric decay, starting at 85° | Same |
| Gaussian variation σ | 10⁻⁵ | Same |
| Temperature | Fixed at one | Same |
| Caption assignment | Sampled uniformly along each leaf's root-to-leaf path | Balanced by cycling along each path |
| Additive prototype shift | Zero | Root norm 10⁻⁴ |
| Regularization λ | 10⁻³ | Full grid in `scripts/run_figure5.sh` |
| Optimizer | L-BFGS, learning rate one | Same; final metrics recomputed with the float64 solver |
| Training/checkpoints | 50 outer steps; checkpoint every step | 500 outer steps; checkpoint every 20 steps |
| Data seed | 42 | 42 |
| Violation summary | Mean of depth-wise means at depths one through 4 | Mean over all eligible non-leaf anchors |

The synthetic metric uses the maximum descendant angle, without the real-data percentile relaxation. Prototype alignment is the minimum cosine similarity between a non-root learned text embedding and its image prototype. The convergence panel plots InfoNCE loss without the quadratic regularizer.

The implementation averages InfoNCE terms over images. Its regularization coefficient is the plotted λ. For a loss written as a sum over N images, the equivalent coefficient is λ_sum = N × λ_code.
