"""Extract ImageNet features and reproduce the real-data HAS measurements."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

from .hierarchy import build_imagenet_hierarchy_graph, collapse_unary_nodes
from .measurement import test_branch_separation
from .splits import (aggregate_descendant_leaf_embeddings, average_branch_separation_results,
                     shuffle_leaf_assignments, _gaussian_splits_from_counts)

ROOT = Path(__file__).resolve().parents[1]


def image_files(directory, synset, seed=42):
    files = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    material = f'{seed}:{synset}'.encode()
    random.Random(int(hashlib.sha256(material).hexdigest()[:16], 16)).shuffle(files)
    return files


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_graph(path):
    graph, names = build_imagenet_hierarchy_graph(path)
    leaves = {n for n in graph if graph.out_degree(n) == 0}
    collapsed, _ = collapse_unary_nodes(graph, protected_nodes=leaves)
    return graph, collapsed, names


def measure_splits(graph, a, b, model):
    forward = test_branch_separation(graph, a, model, embedding_type='image',
                                    comparison_embeddings=b, cone_boundary_percentile=95)
    reverse = test_branch_separation(graph, b, model, embedding_type='image',
                                    comparison_embeddings=a, cone_boundary_percentile=95)
    return average_branch_separation_results(forward, reverse, model)


def extract(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    from .imagenet_dic import IMAGENET_TEMPLATES
    from .imagenet_synset_map import SYNSET_TO_IMAGENET_CLASSES_CLIP_NAMES

    graph, _, names = load_graph(args.hierarchy)
    leaves = sorted(n for n in graph if graph.out_degree(n) == 0)
    output = args.cache_root / f'{args.model}_seed{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=args.model, seed=args.seed, image_order_seed=args.image_order_seed,
                  max_images=args.max_images, hierarchy_sha256=hashlib.sha256(args.hierarchy.read_bytes()).hexdigest(),
                  imagenet_root=str(args.imagenet_root.resolve()), accumulation='float32')
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text()) != config:
        raise ValueError(f'Incompatible extraction configuration in {output}; choose another cache root.')
    write_json(output/'config.json', config)
    torch.set_num_threads(args.threads)
    model = None
    if args.model == 'raw_pixels':
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif args.model == 'openai':
        import clip
        model, preprocess = clip.load('ViT-L/14', device=args.device)
        tokenizer = lambda texts: clip.tokenize(texts, truncate=True)
    else:
        import open_clip
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        pretrained = 'datacomp_xl_s13b_b90k' if args.model == 'datacomp1b' else None
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained=pretrained)
        model = model.to(args.device)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    if model is not None:
        model.eval()
    if not args.text_only:
        for leaf in tqdm(leaves, desc='ImageNet leaves'):
            files = image_files(args.imagenet_root/'train'/leaf, leaf, args.image_order_seed)
            if args.max_images > 0:
                files = files[:args.max_images]
            if len(files) < 2:
                raise ValueError(f'{leaf} needs at least two images for disjoint splits.')
            target = output/f'{leaf}.npz'
            if target.exists():
                continue
            sums = [None, None]
            counts = [0, 0]
            for start in range(0, len(files), args.batch_size):
                images = []
                for file in files[start:start+args.batch_size]:
                    with Image.open(file) as image:
                        images.append(preprocess(image.convert('RGB')))
                batch = torch.stack(images).to(args.device)
                with torch.no_grad():
                    features = model.encode_image(batch) if model is not None else batch.flatten(1)
                    features = F.normalize(features, dim=1).float().cpu().numpy()
                for split in (0, 1):
                    selected = features[(np.arange(len(features))+start) % 2 == split]
                    value = selected.sum(axis=0, dtype=np.float32)
                    sums[split] = value if sums[split] is None else sums[split]+value
                    counts[split] += len(selected)
            np.savez_compressed(target, a=sums[0]/counts[0], b=sums[1]/counts[1],
                                full=(sums[0]+sums[1])/sum(counts), na=counts[0], nb=counts[1])
    if model is not None and not args.skip_text:
        representations = {}
        for node in tqdm(graph, desc='Concept prompts'):
            name = SYNSET_TO_IMAGENET_CLASSES_CLIP_NAMES.get(node, names[node])
            prompts = [template.format(name) for template in IMAGENET_TEMPLATES]
            with torch.no_grad():
                features = F.normalize(model.encode_text(tokenizer(prompts).to(args.device)), dim=1)
            representations[node] = features.float().mean(dim=0).cpu().numpy()
        np.savez_compressed(output/'text.npz', **representations)


def load_leaves(cache, leaves):
    a, b, full, ca, cb = {}, {}, {}, {}, {}
    for leaf in leaves:
        with np.load(cache/f'{leaf}.npz') as data:
            a[leaf], b[leaf], full[leaf] = data['a'], data['b'], data['full']
            ca[leaf], cb[leaf] = int(data['na']), int(data['nb'])
    return a, b, full, ca, cb


def evaluate_splits(graph, splits, model):
    a, b, ca, cb = splits
    a, _ = aggregate_descendant_leaf_embeddings(graph, a, ca)
    b, _ = aggregate_descendant_leaf_embeddings(graph, b, cb)
    return measure_splits(graph, a, b, model)


def figure2(args):
    original, graph, _ = load_graph(args.hierarchy)
    leaves = sorted(n for n in graph if graph.out_degree(n) == 0)
    results = {}
    for model in ['random_vit', 'openai', 'datacomp1b']:
        cache = args.cache_root/f'{model}_seed0'
        with np.load(cache/'text.npz') as data:
            text = {n:data[n] for n in graph}
        _, _, full, ca, cb = load_leaves(cache, leaves)
        counts = {n:ca[n]+cb[n] for n in leaves}
        # Figure 2 uses each distinct descendant leaf once, weighted by image count.
        image = dict(full)
        for node in original:
            if original.out_degree(node) == 0:
                continue
            descendants = sorted(n for n in nx.descendants(original,node) if n in full)
            image[node] = np.average(np.stack([full[n] for n in descendants]), axis=0,
                                     weights=[counts[n] for n in descendants]).astype(np.float32)
        for modality, embeddings in [('text',text),('image',image)]:
            results[f'{modality}_{model}'] = test_branch_separation(
                graph, embeddings, model, embedding_type=modality, cone_boundary_percentile=95)
    write_json(args.output/'measurements.json', results)
    from .plotting import plot_figure2
    plot_figure2(results, args.output)


def table1(args):
    _, graph, _ = load_graph(args.hierarchy)
    leaves = sorted(n for n in graph if graph.out_degree(n) == 0)
    rows = []
    leaf_counts = None
    for model, seeds in [('raw_pixels',[0]),('random_vit',[0,1,2]),('openai',[0])]:
        observed, null = [], []
        for seed in seeds:
            a,b,_,ca,cb = load_leaves(args.cache_root/f'{model}_seed{seed}',leaves)
            if model == 'openai':
                leaf_counts = {n:ca[n]+cb[n] for n in leaves}
            result = evaluate_splits(graph,(a,b,ca,cb),model)
            observed.append(result['aggregated_statistics']['violation_proportion']['mean'])
            for shuffle in range(args.shuffle_seeds):
                splits = shuffle_leaf_assignments(a,b,ca,cb,np.random.default_rng(shuffle))
                result = evaluate_splits(graph,splits,model)
                null.append(result['aggregated_statistics']['violation_proportion']['mean'])
        rows.append(dict(model=model,observed=observed,null=null))
    observed, null = [], []
    for seed in range(args.gaussian_seeds):
        splits = _gaussian_splits_from_counts(leaf_counts,seed,dim=768,normalize=True)
        result = evaluate_splits(graph,splits,'random_gaussian')
        observed.append(result['aggregated_statistics']['violation_proportion']['mean'])
        if seed < args.gaussian_null_seeds:
            for shuffle in range(args.shuffle_seeds):
                permuted = shuffle_leaf_assignments(*splits,np.random.default_rng(shuffle))
                result = evaluate_splits(graph,permuted,'random_gaussian')
                null.append(result['aggregated_statistics']['violation_proportion']['mean'])
    rows.append(dict(model='random_gaussian',observed=observed,null=null))
    write_json(args.output/'measurements.json',rows)
    from .plotting import write_table1
    write_table1(rows,args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for command in ['extract','figure2','table1']:
        p = sub.add_parser(command)
        p.add_argument('--hierarchy',type=Path,default=ROOT/'data/imagenet_hierarchy.json')
        p.add_argument('--cache-root',type=Path,default=Path('results/cache'))
        if command != 'extract':
            p.add_argument('--output',type=Path,default=Path('results')/command)
    p = sub.choices['extract']
    p.add_argument('--imagenet-root',type=Path,required=True,help='Directory containing train/<synset>/*.JPEG')
    p.add_argument('--model',choices=['raw_pixels','random_vit','openai','datacomp1b'],required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--image-order-seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-images',type=int,default=-1,help='Use -1 for all training images; positive values are smoke runs')
    p.add_argument('--skip-text',action='store_true')
    p.add_argument('--text-only',action='store_true')
    p = sub.choices['table1']
    p.add_argument('--gaussian-seeds',type=int,default=40)
    p.add_argument('--gaussian-null-seeds',type=int,default=5)
    p.add_argument('--shuffle-seeds',type=int,default=5)
    args = parser.parse_args()
    {'extract':extract,'figure2':figure2,'table1':table1}[args.command](args)


if __name__ == '__main__':
    main()
