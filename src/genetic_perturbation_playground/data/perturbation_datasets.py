import pertpy as pt
import scanpy as sc


def replogle_load_and_preprocess() -> sc.AnnData:
    adata = pt.data.replogle_2022_k562_essential()
    sc.pp.filter_cells(adata, min_counts=100)
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.normalize_total(adata, target_sum=4000.0)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
    return adata[(adata.obs["nperts"] == 1) | (adata.obs["perturbation"] == "control")].copy()
