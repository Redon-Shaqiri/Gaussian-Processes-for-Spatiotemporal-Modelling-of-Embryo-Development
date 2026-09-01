import numpy as np
import scanpy as sc

def genes_variability_ranked(data, n_top_genes=50):
    sc.pp.highly_variable_genes(data, n_top_genes=n_top_genes, flavor="cell_ranger")

    highly_variable_genes = (data.var[data.var.highly_variable].sort_values("dispersions_norm", ascending=False))

    genes = highly_variable_genes.index.tolist()

    return genes

def gene_expression_values(adata, gene):
    y = adata[:, gene].X
    y = np.asarray(y).squeeze()
    return y


def multiple_gene_expression_values(adata, genes):
    Y = adata[:, genes].X
    Y = np.asarray(Y)
    return Y