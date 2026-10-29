#!/usr/bin/env python3
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import tqdm as terminal_tqdm
from scgpt.tasks import embed_data
import SEACells
from SEACells import build_graph
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEGMENT_RELATIONS = {
    "v_gene_a": "uses_v_a",
    "j_gene_a": "uses_j_a",
    "v_gene_b": "uses_v_b",
    "d_gene_b": "uses_d_b",
    "j_gene_b": "uses_j_b",
}
TCR_COLUMNS = ["cdr3_aa_a", "cdr3_aa_b", *SEGMENT_RELATIONS]
METACELL_AUX_COLUMNS = [
    "clonotype_entropy",
    "clonotype_entropy_normalized",
    "dominant_clonotype_fraction",
    "clonotype_richness",
    "n_tcr_cells",
    "tcr_cell_fraction",
]
RELATIONS = [
    "metacell__rna_sim__metacell",
    "metacell__scores__geneset",
    "geneset__rev_scores__metacell",
    "metacell__contains__metatcr",
    "metatcr__rev_contains__metacell",
    "metatcr__shared_trb__metatcr",
]
for relation in SEGMENT_RELATIONS.values():
    RELATIONS.extend(
        [f"metatcr__{relation}__segment", f"segment__rev_{relation}__metatcr"]
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("graphs"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "scGPT_human")
    parser.add_argument("--gmt", type=Path, default=ROOT / "resources" / "MSigDB_Hallmark_2020.gmt")
    parser.add_argument("--segment-vocabulary", type=Path, default=ROOT / "resources" / "segment_vocabulary.csv")
    parser.add_argument("--counts-layer",default="auto",help="auto: counts layer or X; X: use X; otherwise layer name")
    parser.add_argument("--sample-col", default=None, help="Default: sample_id, then biosample_id")
    parser.add_argument("--patient-col", default=None, help="Default: patient_id, patient, then donor_id")
    parser.add_argument("--tcr-mode", choices=["any", "paired"], default="any", help="any: retain at least one chain; paired: require both chains")
    parser.add_argument("--target-size", type=int, default=50)
    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ["target_size", "knn", "batch_size", "threads"]:
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    return args


def clean(series):
    result = series.astype("string").str.strip()
    return result.mask(result.str.upper().isin(["", "NA", "NAN", "NONE", "<NA>"]))


def read_input(args):
    data = ad.read_h5ad(args.input_h5ad)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Cell IDs and gene names must be unique")
    aliases = {
        "sample_id": (
            [args.sample_col] if args.sample_col else ["sample_id", "biosample_id"]
        ),
        "patient_id": (
            [args.patient_col]
            if args.patient_col
            else ["patient_id", "patient", "donor_id"]
        ),
        "cdr3_aa_a": ["cdr3_aa_a", "TRA_cdr3_aa", "TCR_cdr3aa_a"],
        "cdr3_aa_b": ["cdr3_aa_b", "TRB_cdr3_aa", "TCR_cdr3aa_b"],
        "v_gene_a": ["v_gene_a", "TRA_v_gene", "TCR_v_a"],
        "v_gene_b": ["v_gene_b", "TRB_v_gene", "TCR_v_b"],
        "j_gene_a": ["j_gene_a", "TRA_j_gene"],
        "j_gene_b": ["j_gene_b", "TRB_j_gene"],
        "d_gene_b": ["d_gene_b", "TRB_d_gene"],
    }
    for target, candidates in aliases.items():
        source = next((c for c in candidates if c in data.obs), None)
        if source is None and target in ["sample_id", "patient_id"]:
            raise ValueError(
                f"Missing {target}; specify its column with the CLI option"
            )
        data.obs[target] = (
            clean(data.obs[source])
            if source
            else pd.Series(pd.NA, index=data.obs_names, dtype="string")
        )
    a, b = data.obs.cdr3_aa_a.notna(), data.obs.cdr3_aa_b.notna()
    keep = a & b if args.tcr_mode == "paired" else a | b
    print(
        f"TCR filter ({args.tcr_mode}): {data.n_obs} -> {int(keep.sum())}", flush=True
    )
    data = data[keep].copy()
    if not data.n_obs:
        raise ValueError("No cells with usable TCR sequences")
    if data.obs[["patient_id", "sample_id"]].isna().any().any():
        raise ValueError("Retained cells have missing sample/patient IDs")
    if data.obs.groupby("sample_id", observed=True).patient_id.nunique().max() != 1:
        raise ValueError("Each sample must belong to exactly one patient")
    layer = args.counts_layer
    if layer == "auto":
        layer = "counts" if "counts" in data.layers else "X"
    data.X = sp.csr_matrix(
        data.X if layer == "X" else data.layers[layer], dtype=np.float32
    )
    if not np.isfinite(data.X.data).all() or (data.X.data < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if not np.allclose(data.X.data, np.round(data.X.data), atol=1e-6, rtol=0):
        raise ValueError("Expected raw integer RNA counts")
    # Store metadata as object columns for H5AD serialization.
    for col in aliases:
        data.obs[col] = (
            data.obs[col].astype(object).where(data.obs[col].notna(), np.nan)
        )
    print(f"Counts source: {layer}; input {data.shape}", flush=True)
    return data


def make_metacells(data, args):
    build_graph.tqdm = terminal_tqdm
    build_graph.NUM_CORES = args.threads
    embedded = embed_data(
        data,
        model_dir=args.model_dir,
        gene_col="index",
        max_length=1200,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        use_fast_transformer=False,
        return_new_adata=True,
    )
    x = np.asarray(embedded.X, dtype=np.float32)
    if x.shape != (data.n_obs, 512) or not np.isfinite(x).all():
        raise ValueError(f"Invalid scGPT output: {x.shape}")
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    assignments = np.empty(data.n_obs, dtype=object)
    samples = data.obs.sample_id.to_numpy()
    logs = []
    for i, sample in enumerate(pd.unique(samples)):
        np.random.seed(args.seed + i)
        positions = np.flatnonzero(samples == sample)
        n = len(positions)
        requested = max(1, min(n, round(n / args.target_size)))
        local = ad.AnnData(obs=data.obs.iloc[positions].copy())
        local.obsm["X_scGPT"] = x[positions]
        status = "single_metacell"
        labels = np.full(n, "0")
        if requested > 1:
            with threadpool_limits(limits=args.threads):
                model = SEACells.core.SEACells(
                    local,
                    build_kernel_on="X_scGPT",
                    n_SEACells=requested,
                    use_gpu=False,
                    verbose=False,
                    n_waypoint_eigs=min(10, max(2, requested), n - 1),
                    n_neighbors=min(30, n - 1),
                    convergence_epsilon=1e-3,
                    max_franke_wolfe_iters=50,
                )
                print(f"{sample}: constructing SEACells kernel", flush=True)
                model.construct_kernel_matrix()
                print(f"{sample}: initializing {requested} SEACells archetypes", flush=True)
                model.initialize_archetypes()
                try:
                    print(f"{sample}: fitting SEACells", flush=True)
                    model.fit(max_iter=100, min_iter=10)
                    status = "converged"
                except RuntimeWarning as warning:
                    if "SEACell" not in local.obs:
                        raise
                    status = f"not_converged: {warning}"
                labels = local.obs["SEACell"].astype(str).to_numpy()
        assignments[positions] = [f"{sample}::MC::{v}" for v in labels]
        logs.append(
            dict(
                sample_id=sample,
                n_cells=n,
                requested_metacells=requested,
                observed_metacells=len(set(labels)),
                status=status,
            )
        )
        print(f"{sample}: {n} cells -> {len(set(labels))} metacells", flush=True)
    codes, labels = pd.factorize(assignments, sort=True)
    membership = sp.csr_matrix(
        (np.ones(data.n_obs), (codes, np.arange(data.n_obs))),
        shape=(len(labels), data.n_obs),
    )
    sizes = np.bincount(codes)
    features = np.asarray(membership @ x, dtype=np.float32) / sizes[:, None]
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    mapping = data.obs[["patient_id", "sample_id"]].copy()
    mapping.insert(0, "cell_id", data.obs_names)
    mapping["metacell_id"], mapping["metacell_label"] = codes, assignments
    obs = mapping.groupby("metacell_id", sort=True)[
        ["patient_id", "sample_id", "metacell_label"]
    ].first()
    obs["n_cells"] = sizes
    obs["graph_id"] = obs["sample_name"] = obs.sample_id
    obs.index = obs.index.astype(str)
    mc = ad.AnnData(
        (membership @ data.X).astype(np.float32).tocsr(), obs=obs, var=data.var.copy()
    )
    mc.obsm["X_scGPT"] = features.astype(np.float32)
    mc.uns["construction"] = dict(
        sample_local=True, target_cells_per_metacell=args.target_size, labels_used=False
    )
    pd.DataFrame(logs).to_csv(args.output_dir / "seacells_qc.csv", index=False)
    mapping.to_csv(args.output_dir / "cell_to_metacell_with_id.csv", index=False)
    return mc, mapping


def score_hallmark(mc, args):
    import scanpy as sc

    sets = {}
    for line in args.gmt.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            sets[fields[0]] = fields[2:]
    if len(sets) != 50:
        raise ValueError("Expected exactly 50 Hallmark gene sets")
    norm = mc.copy()
    sc.pp.normalize_total(norm, target_sum=1e4)
    sc.pp.log1p(norm)
    lookup = {g.upper(): g for g in norm.var_names}
    raw = np.zeros((mc.n_obs, 50), dtype=np.float32)
    qc = []
    for j, (name, genes) in enumerate(sets.items()):
        matched = [lookup[g.upper()] for g in genes if g.upper() in lookup]
        if len(matched) < 5:
            raise ValueError(
                f"{name}: only {len(matched)} matched genes; use gene symbols"
            )
        raw[:, j] = np.asarray(norm[:, matched].X.mean(axis=1)).ravel()
        qc.append(
            dict(pathway=name, genes_in_set=len(genes), genes_matched=len(matched))
        )
    mean, std = raw.mean(0), np.maximum(raw.std(0), 1e-6)
    mc.obsm["X_hallmark_raw"] = raw
    mc.obsm["X_hallmark"] = (raw - mean) / std
    mc.uns["hallmark_names"] = list(sets)
    mc.uns["hallmark_scoring"] = dict(
        standardization="input cohort mean/std", mean=mean, std=std
    )
    pd.DataFrame(qc).to_csv(args.output_dir / "hallmark_gene_match_qc.csv", index=False)


def make_metatcr(data, mc, mapping, args):
    import sceptr
    import tidytcells as tt

    obs = data.obs[["patient_id", "sample_id", *TCR_COLUMNS]].copy()
    identity = (
        obs[["sample_id", "cdr3_aa_a", "cdr3_aa_b"]].fillna("").apply(tuple, axis=1)
    )
    obs["metatcr_id"], _ = pd.factorize(identity, sort=True)
    obs["paired_cdr3"] = obs.cdr3_aa_a.fillna("") + "|" + obs.cdr3_aa_b.fillna("")
    obs["metatcr_label"] = obs.sample_id + "::TCR::" + obs.paired_cdr3
    obs["patient_clonotype_id"] = obs.patient_id + "::TCR::" + obs.paired_cdr3
    grouped = obs.groupby("metatcr_id", sort=True)
    meta = grouped.first()
    meta["n_cells"] = grouped.size()
    meta["metatcr_id"] = meta.index
    meta["graph_id"] = meta["sample_name"] = meta.sample_id
    mapping = mapping.copy()
    mapping["metatcr_id"] = obs.metatcr_id.to_numpy()
    mapping[["cell_id", "sample_id", "metatcr_id"]].to_csv(
        args.output_dir / "cell_to_metatcr.csv", index=False
    )
    counts = (
        mapping.groupby(["metacell_id", "metatcr_id"]).size().rename("n").reset_index()
    )
    counts["p"] = counts.n / counts.groupby("metacell_id").n.transform("sum")
    counts["entropy"] = -counts.p * np.log(counts.p)
    stats = counts.groupby("metacell_id").agg(
        clonotype_entropy=("entropy", "sum"),
        clonotype_richness=("metatcr_id", "nunique"),
        dominant_clonotype_fraction=("p", "max"),
        n_tcr_cells=("n", "sum"),
    )
    stats["clonotype_entropy_normalized"] = stats.clonotype_entropy / np.log(
        stats.clonotype_richness.clip(lower=2)
    )
    stats = stats.reindex(mc.obs_names.astype(int))
    for col in stats:
        mc.obs[col] = stats[col].to_numpy()
    mc.obs["tcr_cell_fraction"] = mc.obs.n_tcr_cells / mc.obs.n_cells
    x = meta[
        ["v_gene_a", "cdr3_aa_a", "j_gene_a", "v_gene_b", "cdr3_aa_b", "j_gene_b"]
    ].copy()
    x.columns = ["TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"]
    x = x.where(x.notna(), None)
    for col in ["TRAV", "TRAJ", "TRBV", "TRBJ"]:
        x[col] = x[col].map(
            lambda v: (
                tt.tr.standardize(
                    str(v).split(",")[0],
                    species="homosapiens",
                    precision="allele",
                    enforce_functional=True,
                    on_fail="reject",
                    log_failures=False,
                )
                if v
                else None
            )
        )
    full = x[["TRAV", "TRAJ", "TRBV", "TRBJ"]].notna().all(axis=1).to_numpy()
    embedding = np.empty((len(meta), 64), dtype=np.float32)
    for mask, factory, columns in [
        (full, sceptr.variant.default, list(x)),
        (~full, sceptr.variant.cdr3_only, ["CDR3A", "CDR3B"]),
    ]:
        if not mask.any():
            continue
        model = factory()
        if args.device.startswith("cuda"):
            model.enable_hardware_acceleration()
        model.set_batch_size(args.batch_size)
        embedding[mask] = model.calc_vector_representations(
            x.loc[mask, columns].reset_index(drop=True)
        )
    if embedding.shape != (len(meta), 64) or not np.isfinite(embedding).all():
        raise ValueError("Invalid SCEPTR embeddings")
    meta["sceptr_full_model"] = full
    meta.index = pd.Index(meta.metatcr_id.astype(str), name="metatcr_node")
    for col in TCR_COLUMNS:
        meta[col] = pd.Categorical(meta[col])

    mt = ad.AnnData(
        embedding,
        obs=meta,
        var=pd.DataFrame(index=[f"SCEPTR_{i:02d}" for i in range(64)]),
    )
    return mt, mapping


def minmax_weight(values):
    values = np.asarray(values, np.float32)
    if not values.size:
        return values
    low, high = float(values.min()), float(values.max())
    return np.ones_like(values) if high - low < 1e-8 else (values - low) / (high - low)


def segment_calls(value):
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>", "na"}:
        return []
    text = text.replace(";", ",").replace("|", ",")
    return sorted(
        {
            x.strip().split("*")[0].replace("/DV", "DV")
            for x in text.split(",")
            if x.strip()
        }
    )


def empty_edge():
    return torch.empty((2, 0), dtype=torch.long)


def graph_filename(graph_id):
    return f"graph_{hashlib.sha1(str(graph_id).encode()).hexdigest()[:12]}.pt"


def build_graphs(mc, mt, cell_map, args):
    OUTPUT_DIR = GRAPH_INPUT_DIR = GRAPH_DIR = args.output_dir
    SEGMENT_VOCABULARY = args.segment_vocabulary
    METACELL_KNN = args.knn
    vocabulary = pd.read_csv(args.segment_vocabulary).sort_values("segment_id")
    segment_ids = vocabulary.segment_id.to_numpy()
    if not np.array_equal(segment_ids, np.arange(len(vocabulary))):
        raise ValueError("Segment IDs must be contiguous from zero")
    SEGMENT_LABELS = vocabulary.segment_label.astype(str).tolist()
    if len(set(SEGMENT_LABELS)) != len(SEGMENT_LABELS):
        raise ValueError("Duplicate segment labels")
    SEGMENT_TO_ID = dict(zip(SEGMENT_LABELS, segment_ids))
    for field in SEGMENT_RELATIONS:
        if f"{field}:<UNK>" not in SEGMENT_TO_ID:
            raise ValueError(f"Missing unknown segment for {field}")
    INPUT_DIMS = dict(metacell=518, metatcr=64, geneset=50, segment=len(vocabulary))
    identity_geneset, identity_segment = (torch.eye(50), torch.eye(len(vocabulary)))
    mc_obs = mc.obs.copy()
    mc_obs["global_metacell_id"] = mc.obs_names.astype(int)
    mc_obs = mc_obs.reset_index(drop=True)
    mt_obs = mt.obs.reset_index(drop=True).copy()
    mc_aux = mc_obs[METACELL_AUX_COLUMNS].to_numpy(np.float32)
    for col in ["clonotype_richness", "n_tcr_cells"]:
        j = METACELL_AUX_COLUMNS.index(col)
        mc_aux[:, j] = np.log1p(mc_aux[:, j])
    mc_features = np.concatenate([mc.obsm["X_scGPT"], mc_aux], axis=1).astype(
        np.float32
    )
    mc_hallmark = np.asarray(mc.obsm["X_hallmark"], dtype=np.float32)
    mt_sceptr = np.asarray(mt.X, dtype=np.float32)
    hallmark_names = list(mc.uns["hallmark_names"])
    for values in [mc_features, mc_hallmark, mt_sceptr]:
        if not np.isfinite(values).all():
            raise ValueError("Non-finite graph features")
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    rows, unknown = ([], set())
    groups = mc_obs.groupby("sample_id", observed=True, sort=True)
    for graph_number, (graph_id, local_mc) in enumerate(groups, 1):
        graph_id = str(graph_id)
        local_mc = local_mc.copy().sort_values("global_metacell_id")
        local_mc["row"] = local_mc.index.to_numpy(np.int64)
        local_mc = local_mc.reset_index(drop=True)
        local_mt = (
            mt_obs.loc[mt_obs.sample_id.astype(str).eq(graph_id)]
            .copy()
            .sort_values("metatcr_id")
        )
        local_mt["row"] = local_mt.index.to_numpy(np.int64)
        local_mt = local_mt.reset_index(drop=True)
        cells = cell_map.loc[cell_map.sample_id.eq(graph_id)].copy()
        patients = pd.unique(local_mc.patient_id.astype(str))
        if len(patients) != 1 or set(local_mt.patient_id.astype(str)) != {patients[0]}:
            raise AssertionError(f"{graph_id}: graph crosses patients")
        if local_mt.empty or cells.empty:
            raise ValueError(f"{graph_id}: empty metatcr nodes or mapping")
        mc_lut = {int(x): i for i, x in enumerate(local_mc.global_metacell_id)}
        mt_lut = {int(x): i for i, x in enumerate(local_mt.metatcr_id)}
        cells["mc"] = cells.metacell_id.map(mc_lut)
        cells["mt"] = cells.metatcr_id.map(mt_lut)
        if cells[["mc", "mt"]].isna().any().any():
            raise AssertionError(f"{graph_id}: mapping references another sample")
        cells[["mc", "mt"]] = cells[["mc", "mt"]].astype(np.int64)
        mc_rows = local_mc.row.to_numpy(np.int64)
        mt_rows = local_mt.row.to_numpy(np.int64)
        metacell_x = torch.from_numpy(mc_features[mc_rows].copy())
        metatcr_x = torch.from_numpy(mt_sceptr[mt_rows].copy())
        hallmark = mc_hallmark[mc_rows]
        n_mc, n_mt = (len(local_mc), len(local_mt))
        if n_mc > 1:
            z = F.normalize(metacell_x[:, :512], p=2, dim=1)
            similarity = z @ z.T
            similarity.fill_diagonal_(-torch.inf)
            k = min(METACELL_KNN, n_mc - 1)
            values, neighbors = torch.topk(similarity, k=k, dim=1)
            source = torch.arange(n_mc).repeat_interleave(k)
            rna_edge = torch.stack([source, neighbors.ravel()]).long()
            rna_raw = values.ravel().numpy().astype(np.float32)
        else:
            rna_edge, rna_raw = (empty_edge(), np.empty(0, np.float32))
        geneset_edge = torch.from_numpy(
            np.vstack(
                [
                    np.repeat(np.arange(n_mc, dtype=np.int64), len(hallmark_names)),
                    np.tile(np.arange(len(hallmark_names), dtype=np.int64), n_mc),
                ]
            )
        )
        geneset_raw = hallmark.ravel().astype(np.float32)
        pairs = (
            cells.groupby(["mc", "mt"], observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        mc_id, mt_id = (pairs.mc.to_numpy(np.int64), pairs.mt.to_numpy(np.int64))
        mc_size = (
            cells.groupby("mc", observed=True)
            .size()
            .reindex(range(n_mc), fill_value=0)
            .to_numpy(np.float32)
        )
        mt_size = (
            cells.groupby("mt", observed=True)
            .size()
            .reindex(range(n_mt), fill_value=0)
            .to_numpy(np.float32)
        )
        if (mc_size == 0).any() or (mt_size == 0).any():
            raise ValueError(f"{graph_id}: unmapped node")
        membership_edge = torch.from_numpy(np.vstack([mc_id, mt_id]))
        membership_weight = torch.from_numpy(
            (
                pairs["count"].to_numpy(np.float32)
                / np.sqrt(mc_size[mc_id] * mt_size[mt_id])
            ).astype(np.float32)
        )
        ss, st, sw = ([], [], [])
        for _, group in local_mt.groupby("cdr3_aa_b", observed=True):
            nodes = group.index.to_numpy(np.int64)
            for source_node in nodes:
                for target_node in nodes:
                    if source_node != target_node:
                        ss.append(int(source_node))
                        st.append(int(target_node))
                        sw.append(float(mt_size[nodes].sum()))
        if ss:
            shared_edge = torch.tensor([ss, st], dtype=torch.long)
            shared_weight = torch.from_numpy(
                minmax_weight(np.log1p(np.asarray(sw, np.float32)))
            )
        else:
            shared_edge = empty_edge()
            shared_weight = torch.empty(0, dtype=torch.float32)
        segment_edges = {}
        for column, relation in SEGMENT_RELATIONS.items():
            calls = []
            for local_id, value in enumerate(local_mt[column]):
                for gene in segment_calls(value):
                    label = f"{column}:{gene}"
                    if label not in SEGMENT_TO_ID:
                        unknown.add(label)
                        label = f"{column}:<UNK>"
                    calls.append((local_id, SEGMENT_TO_ID[label]))
            segment_edges[relation] = (
                torch.tensor(calls, dtype=torch.long).T if calls else empty_edge()
            )
        edge_index = {
            "metacell__rna_sim__metacell": rna_edge,
            "metacell__scores__geneset": geneset_edge,
            "geneset__rev_scores__metacell": geneset_edge[[1, 0]],
            "metacell__contains__metatcr": membership_edge,
            "metatcr__rev_contains__metacell": membership_edge[[1, 0]],
            "metatcr__shared_trb__metatcr": shared_edge,
        }
        hallmark_weight = torch.sigmoid(torch.from_numpy(geneset_raw.copy()))
        edge_weight = {
            "metacell__rna_sim__metacell": torch.from_numpy(minmax_weight(rna_raw)),
            "metacell__scores__geneset": hallmark_weight,
            "geneset__rev_scores__metacell": hallmark_weight.clone(),
            "metacell__contains__metatcr": membership_weight,
            "metatcr__rev_contains__metacell": membership_weight.clone(),
            "metatcr__shared_trb__metatcr": shared_weight,
        }
        for relation, edge in segment_edges.items():
            forward = f"metatcr__{relation}__segment"
            reverse = f"segment__rev_{relation}__metatcr"
            edge_index[forward], edge_index[reverse] = (edge, edge[[1, 0]])
            edge_weight[forward] = torch.ones(edge.shape[1])
            edge_weight[reverse] = torch.ones(edge.shape[1])
        edge_index = {key: edge_index[key] for key in RELATIONS}
        edge_weight = {key: edge_weight[key] for key in RELATIONS}
        graph = {
            "node_features": {
                "metacell": metacell_x,
                "metatcr": metatcr_x,
                "geneset": identity_geneset,
                "segment": identity_segment,
            },
            "num_nodes": {
                "metacell": n_mc,
                "metatcr": n_mt,
                "geneset": 50,
                "segment": len(SEGMENT_LABELS),
            },
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "raw_edge_weight": {
                "metacell__scores__geneset": torch.from_numpy(geneset_raw.copy()),
                "geneset__rev_scores__metacell": torch.from_numpy(geneset_raw.copy()),
            },
            "node_aux": {},
            "mappings": {
                "metacell_global_ids": local_mc.global_metacell_id.astype(int).tolist(),
                "metatcr_global_ids": local_mt.metatcr_id.astype(int).tolist(),
                "metatcr_labels": local_mt.metatcr_label.astype(str).tolist(),
                "geneset_names": hallmark_names,
                "segment_labels": SEGMENT_LABELS,
            },
            "config": {
                "dataset": OUTPUT_DIR.name,
                "graph_id": graph_id,
                "patient_id": str(patients[0]),
                "metacell_aux_columns": METACELL_AUX_COLUMNS,
                "segment_vocabulary_csv": str(SEGMENT_VOCABULARY),
            },
        }
        filename = graph_filename(graph_id)
        torch.save(graph, GRAPH_DIR / filename)
        rows.append(
            {
                "graph_id": graph_id,
                "patient_id": str(patients[0]),
                "filename": filename,
                "n_metacell": n_mc,
                "n_metatcr": n_mt,
                "n_geneset": 50,
                "n_segment": len(SEGMENT_LABELS),
                **{f"edges_{key}": int(edge_index[key].shape[1]) for key in RELATIONS},
            }
        )
        print(f"built {graph_number}/{len(groups)} ({graph_id})")
    manifest = pd.DataFrame(rows).sort_values("graph_id").reset_index(drop=True)
    manifest.to_csv(GRAPH_INPUT_DIR / "graph_manifest.csv", index=False)
    pd.DataFrame({"unknown_segment_call": sorted(unknown)}).to_csv(
        GRAPH_INPUT_DIR / "unknown_segment_calls.csv", index=False
    )
    provenance = {
        "dataset": OUTPUT_DIR.name,
        "output_graphs": str(GRAPH_DIR),
        "input_dims": INPUT_DIMS,
        "relations": RELATIONS,
        "metacell_features": {
            "scgpt_dimension": 512,
            "auxiliary_columns": METACELL_AUX_COLUMNS,
            "total_dimension": 518,
        },
        "segment_vocabulary_csv": str(SEGMENT_VOCABULARY),
        "graphs": len(manifest),
        "metatcr_nodes": int(manifest.n_metatcr.sum()),
    }
    with (GRAPH_INPUT_DIR / "graph_provenance.json").open("w") as handle:
        json.dump(provenance, handle, indent=2)
    return manifest


def main():
    args = parse_args()
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(args.threads)
    for path in [args.input_h5ad, args.model_dir, args.gmt, args.segment_vocabulary]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")
    data = read_input(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    mc, mapping = make_metacells(data, args)
    score_hallmark(mc, args)
    mt, mapping = make_metatcr(data, mc, mapping, args)
    mc.write_h5ad(args.output_dir / "metacell.h5ad", compression="gzip")
    mt.write_h5ad(args.output_dir / "metatcr.h5ad", compression="gzip")
    mt.obs.to_csv(args.output_dir / "metatcr_metadata.csv", index=False)
    print("Start creating graphs...", flush=True)
    manifest = build_graphs(mc, mt, mapping, args)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    print(f"Saved {len(manifest)} graphs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
