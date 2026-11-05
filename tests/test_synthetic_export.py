import importlib.util
import sqlite3
from pathlib import Path


def test_sqlite_parameters_keep_each_run_separate(tmp_path):
    path=Path('synthetic/export_results.py')
    assert path.exists(), 'SQLite export for the copied solver is missing'
    spec=importlib.util.spec_from_file_location('export_results',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    db=tmp_path/'mlflow.db'
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE params (run_uuid TEXT, key TEXT, value TEXT)')
        conn.executemany('INSERT INTO params VALUES (?,?,?)',[
            ('a','lambda_reg','1e-3'),('a','caption_mode','balanced'),
            ('b','lambda_reg','1e-2'),('b','caption_mode','uniform')])
    assert module.read_run_params(db)=={'a':{'lambda_reg':'1e-3','caption_mode':'balanced'},
                                       'b':{'lambda_reg':'1e-2','caption_mode':'uniform'}}
