"""Export Figure 5 inputs from fresh MLflow SQLite runs and saved synthetic data."""
import argparse
import json
import sqlite3
from pathlib import Path


def read_run_params(database):
    runs = {}
    with sqlite3.connect(database) as connection:
        for run_id, key, value in connection.execute('SELECT run_uuid, key, value FROM params'):
            runs.setdefault(run_id,{})[key] = value
    return runs


def main():
    from compute_branch_depth_margin import _build_single_summary
    from plot_loss_violation import _fetch_trace
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=Path('results/synthetic'))
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--max-iter',type=int,default=100)
    args = parser.parse_args()
    output = args.run_root/'measurements'
    output.mkdir(parents=True,exist_ok=True)
    trace_db = args.run_root/'trace/mlflow.db'
    runs = read_run_params(trace_db)
    matching = [key for key,p in runs.items() if float(p['lambda_reg'])==1e-3]
    if len(matching)!=1:
        raise ValueError('Use a fresh output directory; expected exactly one trace run at lambda=1e-3.')
    trace = _fetch_trace(trace_db,matching[0])
    if not trace:
        raise ValueError('Trace is missing loss or per-depth measurements.')
    (output/'figure5_trace.json').write_text(json.dumps({'source_run_id':matching[0],'trace':trace},indent=2)+'\n')
    rows = []
    for run_id,p in read_run_params(args.run_root/'sweep/mlflow.db').items():
        summary = _build_single_summary(
            strict_data_path=str(args.run_root/'sweep/strict_data'/run_id/'strict_data.npz'),
            depth=int(p['depth']),branching_factor=int(p['branching_factor']),
            caption_mode=p['caption_mode'],temperature=float(p['temperature']),seed=int(p['seed']),
            lambda_reg=float(p['lambda_reg']),max_iter=args.max_iter,num_threads=args.threads,
            loss_direction=p['loss_direction'])
        rows.append(dict(run_id=run_id,lambda_reg=float(p['lambda_reg']),shift=float(p['prototype_shift_norm']),
                         hier_violation_mean=summary['hierarchical_violation']['node_weighted_mean'],
                         prototype_align_min_nonroot=summary['prototype_alignment']['global_min_nonroot'],
                         post_loss=summary['post_loss'],grad_norm=summary['grad_norm'],
                         num_closures=summary['num_closures']))
    rows.sort(key=lambda row:row['lambda_reg'])
    (output/'figure5_sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f'Exported Figure 5 inputs to {output}')


if __name__=='__main__':
    main()
