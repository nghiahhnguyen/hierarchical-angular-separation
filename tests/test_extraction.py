import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_raw_pixels_stream_into_disjoint_cache(tmp_path):
    from has.real_data import extract,load_graph,load_leaves
    leaves=[dict(id=f'n{i:08d}',name=f'leaf{i}',index=i) for i in range(4)]
    tree=dict(id='fall11',name='ImageNet',children=[dict(id='r',name='entity',children=[
        dict(id='a',name='a',children=leaves[:2]),dict(id='b',name='b',children=leaves[2:])])])
    hierarchy=tmp_path/'hierarchy.json';hierarchy.write_text(json.dumps(tree))
    rng=np.random.default_rng(7)
    for leaf in leaves:
        folder=tmp_path/'images/train'/leaf['id'];folder.mkdir(parents=True)
        for j in range(4):
            Image.fromarray(rng.integers(0,256,(24,24,3),dtype=np.uint8)).save(folder/f'{j}.JPEG')
    args=argparse.Namespace(hierarchy=hierarchy,cache_root=tmp_path/'cache',model='raw_pixels',seed=0,
        image_order_seed=42,max_images=-1,imagenet_root=tmp_path/'images',threads=1,device='cpu',
        text_only=False,skip_text=True,batch_size=3)
    extract(args)
    a,b,full,ca,cb=load_leaves(args.cache_root/'raw_pixels_seed0',[leaf['id'] for leaf in leaves])
    for leaf in a:
        assert ca[leaf]==cb[leaf]==2
        np.testing.assert_allclose(full[leaf],(a[leaf]+b[leaf])/2,atol=1e-7)
        assert np.isfinite(full[leaf]).all()
    args.image_order_seed=0
    with pytest.raises(ValueError,match='Incompatible extraction configuration'):
        extract(args)
