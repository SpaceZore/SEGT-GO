import warnings
from sklearn.metrics import average_precision_score as aupr
import torch
import numpy as np
import scipy.sparse as sp
import dgl
from pathlib import Path
from collections import defaultdict
from Bio.Blast.Applications import NcbipsiblastCommandline
from Bio.Blast import NCBIXML
from tqdm import tqdm, trange


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1 / 2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_batch(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct


def fmax(targets: sp.csr_matrix, scores: np.ndarray):
    fmax_ = 0.0, 0.0
    for cut in (c / 100 for c in range(101)):
        cut_sc = sp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
        if np.isnan(p):
            continue
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
        except ZeroDivisionError:
            pass
    return fmax_


def pair_aupr(targets: sp.csr_matrix, scores: np.ndarray, top=200):
    scores[np.arange(scores.shape[0])[:, None],
           scores.argpartition(scores.shape[1] - top)[:, :-top]] = -1e100
    return aupr(targets.toarray().flatten(), scores.flatten())


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    return tensor


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """Convert a torch sparse tensor to a scipy sparse matrix."""

    m_index = torch_sparse._indices().numpy()
    row = m_index[0]
    col = m_index[-1]
    data = torch_sparse._values().numpy()

    sp_matrix = sp.coo_matrix((data, (row, col)), shape=(torch_sparse.size()[0], torch_sparse.size()[-1]))

    return sp_matrix


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    # adjacency_matrix(transpose, scipy_fmt="csr")
    A = g.adj(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    # lap_pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    lap_pos_enc = EigVec[:, 1:pos_enc_dim + 1]

    return lap_pos_enc


def re_features(adj, features, K):
    # 传播之后的特征矩阵,size= (N, 1, K+1, d )
    nodes_features = torch.empty(features.shape[0], 1, K + 1, features.shape[1])

    for i in range(features.shape[0]):
        nodes_features[i, 0, 0, :] = features[i]

    x = features + torch.zeros_like(features)

    for i in range(K):

        x = torch.matmul(adj, x)

        for index in range(features.shape[0]):
            nodes_features[index, 0, i + 1, :] = x[index]

    nodes_features = nodes_features.squeeze()

    return nodes_features


def nor_matrix(adj, a_matrix):
    nor_matrix = torch.mul(adj, a_matrix)
    row_sum = torch.sum(nor_matrix, dim=1, keepdim=True)
    nor_matrix = nor_matrix / row_sum

    return nor_matrix


def psiblast(blastdb, pid_list, fasta_path, output_path: Path, evalue=1e-3, num_iterations=1,
             num_threads=40, bits=True, query_self=False):
    output_path = output_path.with_suffix('.xml')
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cline = NcbipsiblastCommandline(query=fasta_path, db=blastdb, evalue=evalue, outfmt=5, out=output_path,
                                        num_iterations=num_iterations, num_threads=num_threads)
        print(cline)
        cline()
    else:
        print(F'Using exists blast output file {output_path}')
    with open(output_path) as fp:
        psiblast_sim = defaultdict(dict)
        for pid, rec in zip(tqdm(pid_list, desc='Parsing PsiBlast results'), NCBIXML.parse(fp)):
            query_pid, sim = rec.query, []
            assert pid == query_pid
            for alignment in rec.alignments:
                alignment_pid = alignment.hit_def.split()[0]
                if alignment_pid != query_pid or query_self:
                    psiblast_sim[query_pid][alignment_pid] = max(hsp.bits if bits else hsp.identities / rec.query_length
                                                                 for hsp in alignment.hsps)
    return psiblast_sim
