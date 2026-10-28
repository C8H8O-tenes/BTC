"""Convert Lung_0001 sparse RDS to AnnData, aligning metadata by cell ID."""
from pathlib import Path
import subprocess
import tempfile
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent
R_SCRIPT = '/data/hxu10/conda_envs/R440/bin/Rscript'


def main():
    with tempfile.TemporaryDirectory(prefix='lung_rds_') as temporary:
        tmp = Path(temporary)
        code = r'''
        suppressPackageStartupMessages(library(Matrix))
        args <- commandArgs(trailingOnly=TRUE)
        x <- readRDS(args[1])
        stopifnot(inherits(x, "dgCMatrix"))
        stopifnot(!is.null(rownames(x)), !is.null(colnames(x)))
        stopifnot(!anyDuplicated(rownames(x)), !anyDuplicated(colnames(x)))
        writeBin(x@x, file.path(args[2], "values.bin"), size=8, endian="little")
        writeBin(x@i, file.path(args[2], "indices.bin"), size=4, endian="little")
        writeBin(x@p, file.path(args[2], "indptr.bin"), size=4, endian="little")
        writeLines(rownames(x), file.path(args[2], "genes.txt"))
        writeLines(colnames(x), file.path(args[2], "cells.txt"))
        '''
        subprocess.run([R_SCRIPT, '-e', code, str(ROOT / 'Lung_0001.rds'), temporary], check=True)
        genes = pd.Index((tmp / 'genes.txt').read_text().splitlines(), name='gene')
        cells = pd.Index((tmp / 'cells.txt').read_text().splitlines(), name='cell_id')
        # R gene-by-cell CSC has the same arrays as transposed cell-by-gene CSR.
        values = np.fromfile(tmp / 'values.bin', dtype='<f8')
        indices = np.fromfile(tmp / 'indices.bin', dtype='<i4')
        indptr = np.fromfile(tmp / 'indptr.bin', dtype='<i4')
        matrix = csr_matrix((values, indices, indptr), shape=(len(cells), len(genes)))
        matrix.check_format(full_check=True)
    assert cells.is_unique and genes.is_unique
    assert np.isfinite(matrix.data).all()
    metadata = pd.read_csv(ROOT / 'Lung_0001_metadata.txt', sep='\t', index_col='ID')
    assert metadata.index.is_unique and metadata.columns.is_unique
    missing = cells.difference(metadata.index)
    extra = metadata.index.difference(cells)
    if len(missing):
        raise ValueError(f'{len(missing)} expression cells lack metadata: {missing[:5].tolist()}')
    obs = metadata.reindex(cells).copy()
    obs['ID'] = cells.to_numpy()
    a = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
    if {'Axis', 'Axis.x', 'Axis.y'} <= set(obs.columns) and obs['Axis'].eq('UMAP').all():
        coordinates = obs[['Axis.x', 'Axis.y']].to_numpy(dtype=np.float64)
        assert np.isfinite(coordinates).all()
        a.obsm['X_umap'] = coordinates
    a.uns['source_rds'] = 'Lung_0001.rds'
    a.uns['source_metadata'] = 'Lung_0001_metadata.txt'
    a.uns['X_description'] = 'Original RDS expression values, preserved as float64; preprocessing unknown; not asserted to be raw counts.'
    a.uns['metadata_alignment'] = 'Exact cell ID match, reordered to RDS column order.'
    a.uns['metadata_extra_cells'] = len(extra)
    output = ROOT / 'Lung_0001.h5ad'
    a.write_h5ad(output, compression='gzip')
    # Read-back validation: preserve sparse arrays, axis names, metadata and embedding.
    restored = ad.read_h5ad(output)
    assert restored.obs_names.equals(cells) and restored.var_names.equals(genes)
    for part in ['data', 'indices', 'indptr']:
        np.testing.assert_array_equal(getattr(restored.X, part), getattr(matrix, part))
    for column in obs:
        pd.testing.assert_series_equal(restored.obs[column].astype(object), obs[column].astype(object), check_dtype=False)
    if 'X_umap' in a.obsm:
        np.testing.assert_array_equal(restored.obsm['X_umap'], a.obsm['X_umap'])
    print(restored)
    print(f'Metadata matched: {len(cells)}; extra metadata rows: {len(extra)}')
    print(f'Expression range: {matrix.data.min()} .. {matrix.data.max()}; noninteger entries: {np.count_nonzero(matrix.data != np.floor(matrix.data))}')
    print(f'Saved and verified: {output}')


if __name__ == '__main__':
    main()
