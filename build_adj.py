# --------------------------------------------------------
# Part of code borrowed from DeepGraphGO
# --------------------------------------------------------

import click
import numpy as np
import scipy.sparse as ssp
import torch
from tqdm import trange


def get_norm_net_mat(net_mat):
    degree_0 = np.asarray(net_mat.sum(0)).squeeze()
    mat_d_0 = ssp.diags(degree_0 ** -0.5, format='csr')
    degree_1 = np.asarray(net_mat.sum(1)).squeeze()
    mat_d_1 = ssp.diags(degree_1 ** -0.5, format='csr')
    return mat_d_0 @ net_mat @ mat_d_1


@click.command()
@click.argument('ppi_net_mat_path', type=click.Path(exists=True))
@click.argument('adj_path', type=click.Path())
@click.argument('top', type=click.INT, default=100, required=False)
def main(ppi_net_mat_path, adj_path, top):
    ppi_net_mat = (mat_ := ssp.load_npz(ppi_net_mat_path)) + ssp.eye(mat_.shape[0], format='csr')
    print('ppi_net_mat.shape', ppi_net_mat.shape, 'ppi_net_mat.nnz', ppi_net_mat.nnz)
    resource, destination, values = [], [], []
    for i in trange(ppi_net_mat.shape[0]):
        for v_, d_ in sorted(zip(ppi_net_mat[i].data, ppi_net_mat[i].indices), reverse=True)[:top]:
            resource.append(i)
            destination.append(d_)
            values.append(v_)
    ppi_net_mat = get_norm_net_mat(ssp.csc_matrix((values, (resource, destination)), shape=ppi_net_mat.shape).T)
    print('ppi_net_mat.shape', ppi_net_mat.shape, 'ppi_net_mat.nnz', ppi_net_mat.nnz)
    ppi_net_mat_coo = ssp.coo_matrix(ppi_net_mat)
    adj = torch.sparse_coo_tensor([ppi_net_mat_coo.row, ppi_net_mat_coo.col],
                                  ppi_net_mat_coo.data,
                                  size=torch.Size(ppi_net_mat_coo.shape),
                                  dtype=torch.float32)
    torch.save(adj, adj_path)


if __name__ == '__main__':
    main()
