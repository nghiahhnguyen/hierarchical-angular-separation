import json
from pathlib import Path

from has.plotting import plot_figure2, plot_figure5, write_table1


def test_table_uses_sample_spread_only_for_random_vit(tmp_path):
    rows = [dict(model='raw_pixels',observed=[2],null=[1,3]),
            dict(model='random_vit',observed=[0,2],null=[0,2,4]),
            dict(model='openai',observed=[5],null=[1,3]),
            dict(model='random_gaussian',observed=[0,4],null=[2,6])]
    write_table1(rows,tmp_path)
    text=(tmp_path/'table1.md').read_text()
    assert '| Raw pixels | 2.00 | 2.00 ± 1.00 |' in text
    assert '| Randomly initialized ViT-L/14 | 1.00 ± 1.41 | 2.00 ± 2.00 |' in text
    assert '| i.i.d. Gaussian vectors | 2.00 ± 2.00 | 4.00 ± 2.00 |' in text
    assert r'\begin{tabular}' in (tmp_path/'table1.tex').read_text()


def test_figure2_renders_generated_measurements(tmp_path):
    records = [{'depth':depth,'violation_proportion':value}
               for depth in [1,2] for value in [0.0,1.0]]
    measurements = {f'{modality}_{model}':{'per_node_results':records}
                    for modality in ['text','image'] for model in ['random_vit','openai','datacomp1b']}
    plot_figure2(measurements,tmp_path)
    assert (tmp_path/'figure2.png').stat().st_size>0
    assert (tmp_path/'figure2.pdf').stat().st_size>0


def test_figure5_renders_only_generated_measurements(tmp_path):
    inputs=tmp_path/'measurements';inputs.mkdir()
    trace={'trace':[dict(step=step,main_loss=loss,avg_viol=violation)
                    for step,loss,violation in [(0,2.0,0.5),(10,1.0,0.0)]]}
    sweep=[dict(shift=1e-4,lambda_reg=regularization,hier_violation_mean=violation,
                prototype_align_min_nonroot=alignment)
           for regularization,violation,alignment in [(1e-5,0.5,0.3),(1e-3,0.0,0.8)]]
    (inputs/'figure5_trace.json').write_text(json.dumps(trace))
    (inputs/'figure5_sweep.json').write_text(json.dumps(sweep))
    output=tmp_path/'plots'
    plot_figure5(inputs,output)
    for stem in ['figure5a','lambda_vs_hiervio_protoalign_shift_1e-04']:
        assert (output/f'{stem}.png').stat().st_size>0
        assert (output/f'{stem}.pdf').stat().st_size>0
