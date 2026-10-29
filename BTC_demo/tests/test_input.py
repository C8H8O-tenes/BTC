"""Input contract checks; small synthetic matrices test validation only."""
from pathlib import Path
import sys
import tempfile
from argparse import Namespace
import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from creat_graph import read_input

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)/'input.h5ad'
    args = Namespace(input_h5ad=path,sample_col=None,patient_col=None,tcr_mode='any',counts_layer='auto')
    data = ad.AnnData(np.array([[1,2],[3,4],[5,6]],dtype=np.float32),obs=pd.DataFrame({'sample_id':['s']*3,'patient_id':['p']*3,'cdr3_aa_a':['CAVR',None,None],'cdr3_aa_b':[None,'CASS',None]},index=['c1','c2','c3']),var=pd.DataFrame(index=['CD3D','TRAC']))
    data.write_h5ad(path)
    assert read_input(args).obs_names.tolist() == ['c1','c2']
    args.tcr_mode='paired'
    try: read_input(args)
    except ValueError as e: assert 'No cells' in str(e)
    else: raise AssertionError('Accepted no usable cells')
    args.tcr_mode='any'
    for scenario in ['fractional','missing_patient','mixed_patients']:
        invalid=data.copy()
        if scenario=='fractional': invalid.X[0,0]=0.5
        elif scenario=='missing_patient': del invalid.obs['patient_id']
        else:
            invalid.obs['patient_id']=invalid.obs['patient_id'].astype(object)
            invalid.obs.loc['c2','patient_id']='p2'
        invalid.write_h5ad(path)
        try: read_input(args)
        except ValueError: pass
        else: raise AssertionError(f'Accepted {scenario}')
print('PASS: single-chain retention, missing-chain exclusion, paired mode, integer counts, required IDs, sample/patient consistency')
