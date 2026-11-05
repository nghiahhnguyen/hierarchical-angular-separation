import importlib
import numpy as np
import networkx as nx


def test_bidirectional_measurement_matches_independent_angle_count():
    assert importlib.util.find_spec('has.real_data') is not None, 'release entry point is missing'
    real = importlib.import_module('has.real_data')
    assert hasattr(real, 'measure_splits'), 'release entry point is missing'
    g = nx.DiGraph([('r','a'),('r','b'),('a','a1'),('a','a2'),('b','b1'),('b','b2')])
    rng = np.random.default_rng(5)
    a = {n: rng.normal(size=4).astype(np.float32) for n in g}
    b = {n: rng.normal(size=4).astype(np.float32) for n in g}
    expected = []
    for n in ['a','b']:
        vals = []
        for anchor, comparison in [(a,b),(b,a)]:
            axis = anchor[n] / np.linalg.norm(anchor[n])
            desc = list(nx.descendants(g,n))
            off = set(g) - {n} - nx.ancestors(g,n) - set(desc)
            angle = lambda k: np.arccos(np.clip(axis @ (comparison[k]/np.linalg.norm(comparison[k])), -1, 1))
            threshold = np.percentile([angle(k) for k in desc],95)
            vals.append(100*sum(angle(k)<threshold for k in off)/len(off))
        expected.append(np.mean(vals))
    result = real.measure_splits(g,a,b,'toy')
    assert np.isclose(result['aggregated_statistics']['violation_proportion']['mean'],np.mean(expected))


def test_image_order_is_reproducible_and_classes_do_not_share_order(tmp_path):
    assert importlib.util.find_spec('has.real_data') is not None, 'release entry point is missing'
    real = importlib.import_module('has.real_data')
    assert hasattr(real, 'image_files'), 'deterministic image ordering is missing'
    for i in range(10):
        (tmp_path/f'{i}.JPEG').touch()
    first = real.image_files(tmp_path, 'n1', seed=42)
    assert first == real.image_files(tmp_path, 'n1', seed=42)
    assert first != real.image_files(tmp_path, 'n2', seed=42)
    assert set(first) == set(tmp_path.glob('*.JPEG'))


def test_null_moves_embeddings_and_counts_together():
    from has.splits import shuffle_leaf_assignments
    a={str(i):np.array([i]) for i in range(6)}
    ca={str(i):i+2 for i in range(6)}
    b={k:v+10 for k,v in a.items()}; cb={k:v+20 for k,v in ca.items()}
    aa,bb,cca,ccb=shuffle_leaf_assignments(a,b,ca,cb,np.random.default_rng(3))
    assert any(aa[k][0]!=a[k][0] for k in a)
    for k in a:
        assert cca[k]==aa[k][0]+2
        assert ccb[k]==cca[k]+20
        assert bb[k][0]==aa[k][0]+10
