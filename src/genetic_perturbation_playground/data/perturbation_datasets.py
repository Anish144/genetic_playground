import scanpy as sc
import pertpy as pt


def replogle_load_and_preprocess(
    n_top_genes: int = 5000,
    target_sum: float = 4000.0,
    min_counts: int = 100,
    min_cells: int = 5,
) -> sc.AnnData:
    adata = pt.data.replogle_2022_k562_essential()
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    # Ensure perturbed genes are always included even if not highly variable
    perturbed_genes = set(adata.obs["perturbation"].unique()) - {"control"}
    adata.var["highly_variable"] |= adata.var_names.isin(perturbed_genes)
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata[(adata.obs["nperts"] == 1) | (adata.obs["perturbation"] == "control")].copy()