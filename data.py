import pickle
import warnings
from collections import defaultdict

import numpy
import numpy as np
from tqdm import tqdm
import utils
import dgl.data
import torch
import joblib
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
import scipy.sparse as ssp
from utils import psiblast
from Bio import SeqIO, pairwise2

'''
整个data.py文件是用来处理DeepGraphGO数据集的脚本
'''


def get_pid_list(pid_list_file):
    try:
        with open(pid_list_file) as fp:
            return [line.split()[0] for line in fp]
    except TypeError:
        return pid_list_file


def get_go_list(pid_go_file, pid_list):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list := line.split())[0]].append(line_list[1])
        return [pid_go[pid_] for pid_ in pid_list]
    else:
        return None


def get_data(fasta_file, pid_go_file=None):
    pid_list, data_x = [], []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)
        data_x.append(str(seq.seq))
    return pid_list, data_x, get_go_list(pid_go_file, pid_list)


# 获得mlb查询文件，直接把生产的mlb文件放到dataset文件夹就行了，不要每次生成
def get_mlb(mlb_path: Path, labels=None, **kwargs) -> MultiLabelBinarizer:
    # 如果存在直接加载
    if mlb_path.exists():
        return joblib.load(mlb_path)
    # 不存在要调用查询
    mlb = MultiLabelBinarizer(sparse_output=True, **kwargs)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def get_ppi_idx(pid_list, data_y, net_pid_map):
    pid_list_ = tuple(zip(*[(i, pid, net_pid_map[pid])
                            for i, pid in enumerate(pid_list) if pid in net_pid_map]))
    assert pid_list_
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y


def get_homo_ppi_idx(pid_list, fasta_file, data_y, net_pid_map, net_blastdb, blast_output_path):
    blast_sim = psiblast(net_blastdb, pid_list, fasta_file, blast_output_path)
    pid_list_ = []
    for i, pid in enumerate(pid_list):
        blast_sim[pid][None] = float('-inf')
        pid_ = pid if pid in net_pid_map else max(blast_sim[pid].items(), key=lambda x: x[1])[0]
        if pid_ is not None:
            pid_list_.append((i, pid, net_pid_map[pid_]))
    pid_list_ = tuple(zip(*pid_list_))
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y


def get_homoNotSTR_ppi_idx(pid_list, fasta_file, net_pid_map, net_blastdb, blast_output_path):
    blast_sim = psiblast(net_blastdb, pid_list, fasta_file, Path(blast_output_path))
    pid_list_ = []

    for i, pid in enumerate(pid_list):
        blast_sim[pid][None] = float('-inf')
        if pid not in net_pid_map:
            pid_ = max(blast_sim[pid].items(), key=lambda x: x[1])[0]
            if pid_ is not None:
                pid_list_.append((i, pid, net_pid_map[pid_]))

    pid_list_ = tuple(zip(*pid_list_))
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2]


def get_pid_go(pid_go_file):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list := line.split('\t'))[0]].append(line_list[1])
        return dict(pid_go)
    else:
        return None


def get_pid_go_by_proteinCode(pid_go_file, proteinCode):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                line_list = line.split('\t')
                if int(line_list[-1].strip()) == proteinCode:
                    pid_go[line_list[0]].append(line_list[1])
        return dict(pid_go)
    else:
        return None


def get_pid_go_by_STRHOMO(pid_go_file, test_pid_list):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                line_list = line.split('\t')
                if line_list[0].strip() in test_pid_list:
                    pid_go[line_list[0]].append(line_list[1])
        return dict(pid_go)
    else:
        return None


def output_format(pid_list, go_list, sc_mat):
    pid_go_sc = {}
    for pid_, sc_ in zip(pid_list, sc_mat):
        pid_go_sc[pid_] = {}
        for go_, s_ in zip(go_list, sc_):
            if s_ > 0.0:
                pid_go_sc[pid_][go_] = s_
    return pid_go_sc


def get_pid_go_sc(pid_go_sc_file):
    pid_go_sc = defaultdict(dict)
    with open(pid_go_sc_file) as fp:
        for line in fp:
            pid_go_sc[line_list[0]][line_list[1]] = float((line_list := line.split('\t'))[2])
    return dict(pid_go_sc)


def get_pid_go_mat(pid_go, pid_list, go_list):
    go_mapping = {go_: i for i, go_ in enumerate(go_list)}
    r_, c_, d_ = [], [], []
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go:
            for go_ in pid_go[pid_]:
                if go_ in go_mapping:
                    r_.append(i)
                    c_.append(go_mapping[go_])
                    d_.append(1)
    return ssp.csr_matrix((d_, (r_, c_)), shape=(len(pid_list), len(go_list)))


def get_pid_go_sc_mat(pid_go_sc, pid_list, go_list):
    sc_mat = np.zeros((len(pid_list), len(go_list)))
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go_sc:
            for j, go_ in enumerate(go_list):
                sc_mat[i, j] = pid_go_sc[pid_].get(go_, -1e100)
    return sc_mat


# NAG模型需要使用的输入文件按照hops和GO本体域独立存储成文件，供后续训练直接读取使用
def get_protein_dataset(path_root, dataset_name, hops):
    # 路径list
    path_dict = {
        'feature_path': F'{path_root}/ppi_interpro.npz',
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'train_pid_list_file': F'{path_root}/{dataset_name}_train_pid_list.txt',
        'train_fasta_file': F'{path_root}/{dataset_name}_train.fasta',
        'train_pid_go_file': F'{path_root}/{dataset_name}_train_go.txt',
        'valid_pid_list_file': F'{path_root}/{dataset_name}_valid_pid_list.txt',
        'valid_fasta_file': F'{path_root}/{dataset_name}_valid.fasta',
        'valid_pid_go_file': F'{path_root}/{dataset_name}_valid_go.txt',
        'test_pid_list_file': F'{path_root}/{dataset_name}_test_pid_list.txt',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'valid_blast_output_path': Path(F'{path_root}/{dataset_name}-valid-ppi-blast-out'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}

    # 加载数据集中的pid列表及其GO标签列表
    train_pid_list, _, train_go = get_data(fasta_file=path_dict['train_fasta_file'],
                                           pid_go_file=path_dict['train_pid_go_file'])
    valid_pid_list, _, valid_go = get_data(fasta_file=path_dict['valid_fasta_file'],
                                           pid_go_file=path_dict['valid_pid_go_file'])
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])

    # 使用MLB映射GO标签
    mlb = get_mlb(path_dict['mlb_path'], train_go)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_y, valid_y = mlb.transform(train_go).astype(np.float32), mlb.transform(valid_go).astype(np.float32)
    # 将各集合样本与图中节点对应 并给出对应的标签y(mlb映射后6640维的)
    *_, train_ppi, train_y = get_ppi_idx(train_pid_list, train_y, net_pid_map)
    *_, valid_ppi, valid_y = get_homo_ppi_idx(valid_pid_list, path_dict['valid_fasta_file'],
                                              valid_y, net_pid_map, path_dict['net_blastdb_path'],
                                              path_dict['valid_blast_output_path'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    print('Number of Labels:', len(mlb.classes_))
    print('Size of Training Set: ', len(train_ppi))
    print('Size of Validation Set: ', len(valid_ppi))
    print('Size of Test Set: ', len(test_ppi))

    # 加载特征并提取特征
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load, extract and save feature:'):
        if hop == 0:
            features_csr = ssp.load_npz(path_dict['feature_path'])
        else:
            features_csr = ssp.load_npz(F'{path_root}/{hop}-hops-re-features.npz')
        ssp.save_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz', features_csr[train_ppi])
        ssp.save_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz', features_csr[valid_ppi])
        ssp.save_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz', features_csr[test_ppi])
    ssp.save_npz(F'{path_root}/{dataset_name}-train-y.npz', train_y)
    ssp.save_npz(F'{path_root}/{dataset_name}-valid-y.npz', valid_y)


# NAG模型需要使用的输入文件按照hops和GO本体域独立存储成文件，供后续训练直接读取使用，从start_hop处理到end_hop
def get_protein_dataset_from_hops(path_root, dataset_name, start_hop, end_hop):
    # 路径list
    path_dict = {
        'feature_path': F'{path_root}/ppi_interpro.npz',
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'train_pid_list_file': F'{path_root}/{dataset_name}_train_pid_list.txt',
        'train_fasta_file': F'{path_root}/{dataset_name}_train.fasta',
        'train_pid_go_file': F'{path_root}/{dataset_name}_train_go.txt',
        'valid_pid_list_file': F'{path_root}/{dataset_name}_valid_pid_list.txt',
        'valid_fasta_file': F'{path_root}/{dataset_name}_valid.fasta',
        'valid_pid_go_file': F'{path_root}/{dataset_name}_valid_go.txt',
        'test_pid_list_file': F'{path_root}/{dataset_name}_test_pid_list.txt',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'valid_blast_output_path': Path(F'{path_root}/{dataset_name}-valid-ppi-blast-out'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}

    # 加载数据集中的pid列表及其GO标签列表
    train_pid_list, _, train_go = get_data(fasta_file=path_dict['train_fasta_file'],
                                           pid_go_file=path_dict['train_pid_go_file'])
    valid_pid_list, _, valid_go = get_data(fasta_file=path_dict['valid_fasta_file'],
                                           pid_go_file=path_dict['valid_pid_go_file'])
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])

    # 使用MLB映射GO标签
    mlb = get_mlb(path_dict['mlb_path'], train_go)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_y, valid_y = mlb.transform(train_go).astype(np.float32), mlb.transform(valid_go).astype(np.float32)
    # 将各集合样本与图中节点对应 并给出对应的标签y(mlb映射后6640维的)
    *_, train_ppi, train_y = get_ppi_idx(train_pid_list, train_y, net_pid_map)
    *_, valid_ppi, valid_y = get_homo_ppi_idx(valid_pid_list, path_dict['valid_fasta_file'],
                                              valid_y, net_pid_map, path_dict['net_blastdb_path'],
                                              path_dict['valid_blast_output_path'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    print('Number of Labels:', len(mlb.classes_))
    print('Size of Training Set: ', len(train_ppi))
    print('Size of Validation Set: ', len(valid_ppi))
    print('Size of Test Set: ', len(test_ppi))

    # 加载特征
    # 构建特征张量 也是sparse_coo
    # todo 待优化 应该读完一个接着抽取存储 不应该全部读完再抽取
    hops_features_list = []
    for i in tqdm(range(start_hop, end_hop + 1), total=end_hop + 1 - start_hop, desc='loading features_csr'):
        hops_features_list.append(ssp.load_npz(F'{path_root}/{i}-hops-re-features.npz'))
    for hop in tqdm(range(start_hop, end_hop + 1), total=end_hop + 1 - start_hop, desc='extract and save feature'):
        ssp.save_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz',
                     hops_features_list[hop - start_hop][train_ppi])
        ssp.save_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz',
                     hops_features_list[hop - start_hop][valid_ppi])
        ssp.save_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz',
                     hops_features_list[hop - start_hop][test_ppi])


# 加载数据集供NAG模型训练使用，注意 该方法不支持有test集训练的模型使用
def load_protein_dataset(path_root, dataset_name, hops):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    back_dict['train_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-train-y.npz')
    back_dict['valid_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')
    return back_dict


# 加载数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
def load_protein_dataset_with_test(path_root, dataset_name, hops):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    back_dict['train_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-train-y.npz')
    back_dict['valid_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')

    # test过程中需求的内容
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    pid_go = get_pid_go(path_dict['test_pid_go_file'])
    test_back_dict = {
        'test_pid_list': numpy.array(test_pid_list),
        'mlb': mlb,
        'pid_go': pid_go,
        'test_res_idx_': test_res_idx_
    }
    return back_dict, test_back_dict


# 加载数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值
def load_protein_dataset_with_test_filter(path_root, dataset_name, hops, t):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    back_dict['train_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-train-y.npz')
    back_dict['valid_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')
    back_dict['shap_idx'] = np.load(
        F'/data2/experiment_code/wangyansong/NAG_shap_data/DeepGraphGO-{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
        'arr_0']

    # test过程中需求的内容
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    pid_go = get_pid_go(path_dict['test_pid_go_file'])
    test_back_dict = {
        'test_pid_list': numpy.array(test_pid_list),
        'mlb': mlb,
        'pid_go': pid_go,
        'test_res_idx_': test_res_idx_
    }
    return back_dict, test_back_dict


# 加载数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值
# 增加拉普拉斯特征向量
def load_protein_dataset_with_test_filter_addLap(path_root, dataset_name, hops, t):
    back_dict = {
        'train_x': [],
        'train_lap': None,
        'train_y': None,
        'valid_x': [],
        'valid_lap': None,
        'valid_y': None,
        'test_x': [],
        'test_lap': None,
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    back_dict['train_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-train-y.npz')
    back_dict['valid_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')
    back_dict['shap_idx'] = np.load(
        F'/data2/experiment_code/wangyansong/NAG_shap_data/DeepGraphGO-{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
        'arr_0']
    back_dict['train_lap'] = ssp.load_npz(F'{path_root}/{dataset_name}-train-laplacian-features.npz')
    back_dict['valid_lap'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-laplacian-features.npz')
    back_dict['test_lap'] = ssp.load_npz(F'{path_root}/{dataset_name}-test-laplacian-features.npz')

    # test过程中需求的内容
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    pid_go = get_pid_go(path_dict['test_pid_go_file'])
    test_back_dict = {
        'test_pid_list': numpy.array(test_pid_list),
        'mlb': mlb,
        'pid_go': pid_go,
        'test_res_idx_': test_res_idx_
    }
    return back_dict, test_back_dict


# 加载数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值
# 只加载特定物种蛋白质的test GO标签 在test指标计算时只涉及特定的物种
def load_protein_dataset_with_test_filter_specProtein(path_root, dataset_name, hops, t):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    back_dict['train_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-train-y.npz')
    back_dict['valid_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')
    back_dict['shap_idx'] = np.load(
        F'/data2/experiment_code/wangyansong/NAG_shap_data/DeepGraphGO-{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
        'arr_0']

    # test过程中需求的内容
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    pid_go_human = get_pid_go_by_proteinCode(path_dict['test_pid_go_file'], proteinCode=9606)
    pid_go_mouse = get_pid_go_by_proteinCode(path_dict['test_pid_go_file'], proteinCode=10090)
    print(F'Human Protein Num in test: {len(pid_go_human)}')
    print(F'Mouse Protein Num in test: {len(pid_go_mouse)}')
    test_back_dict = {
        'test_pid_list': numpy.array(test_pid_list),
        'mlb': mlb,
        'pid_go_human': pid_go_human,
        'pid_go_mouse': pid_go_mouse,
        'test_res_idx_': test_res_idx_
    }
    return back_dict, test_back_dict


# 加载数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值
# 根据是否在STRING图中加载蛋白质样本标签，
def load_protein_dataset_with_test_filter_STRHOMO(path_root, dataset_name, hops, t):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    back_dict['train_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-train-y.npz')
    back_dict['valid_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')
    back_dict['shap_idx'] = np.load(
        F'/data2/experiment_code/wangyansong/NAG_shap_data/DeepGraphGO-{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
        'arr_0']

    # test过程中需求的内容
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    # 过滤STR
    pid_go_STR = get_pid_go_by_STRHOMO(path_dict['test_pid_go_file'],
                                       [pid for pid in test_pid_list if pid in net_pid_map])
    # 过滤HOMO
    a, test_pid_list_only_HOMO, c = \
        get_homoNotSTR_ppi_idx(pid_list=test_pid_list, fasta_file=path_dict['test_fasta_file'],
                               net_pid_map=net_pid_map, net_blastdb=path_dict['net_blastdb_path'],
                               blast_output_path=path_dict['test_blast_output_path'])
    pid_go_HOMO = get_pid_go_by_STRHOMO(path_dict['test_pid_go_file'], test_pid_list_only_HOMO)
    print(F'STRING Protein Num in test: {len(pid_go_STR)}')
    print(F'HOMO Protein Num in test: {len(pid_go_HOMO)}')
    test_back_dict = {
        'test_pid_list': numpy.array(test_pid_list),
        'mlb': mlb,
        'pid_go_STR': pid_go_STR,
        'pid_go_HOMO': pid_go_HOMO,
        'test_res_idx_': test_res_idx_
    }
    return back_dict, test_back_dict


# 加载数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值
# 只加载特定物种蛋白质的test GO标签 在test指标计算时只涉及特定的物种
# mode参数指定需要加载的训练集类型，有：
# only_human 只加载人类训练样本在人类样本上test，without_human 加载除人类外的其他物种训练样本 在人类样本上test
# only_mouse 只加载老鼠训练样本在老鼠样本上test，without_mouse 加载除老鼠外的其他物种训练样本 在老鼠样本上test
def load_protein_dataset_with_test_filter_specProteinTrain(path_root, dataset_name, hops, t, mode,
                                                           specProteinTrain_path='/data2/experiment_code/wangyansong/NAG_specProtein_subdataset'):
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
    }
    if mode == 'only_human':
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{specProteinTrain_path}/{dataset_name}-{hop}-hops-train-re-features-only-human.npz'))
            back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
            back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
        back_dict['train_y'] = ssp.load_npz(
            F'{specProteinTrain_path}/{dataset_name}_train_y_human.npz')
        pid_go = get_pid_go_by_proteinCode(path_dict['test_pid_go_file'], proteinCode=9606)
    elif mode == 'without_human':
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{specProteinTrain_path}/{dataset_name}-{hop}-hops-train-re-features-without-human.npz'))
            back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
            back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
        back_dict['train_y'] = ssp.load_npz(
            F'{specProteinTrain_path}/{dataset_name}_train_y_without_human.npz')
        pid_go = get_pid_go_by_proteinCode(path_dict['test_pid_go_file'], proteinCode=9606)
    elif mode == 'only_mouse':
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{specProteinTrain_path}/{dataset_name}-{hop}-hops-train-re-features-only-mouse.npz'))
            back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
            back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
        back_dict['train_y'] = ssp.load_npz(
            F'{specProteinTrain_path}/{dataset_name}_train_y_mouse.npz')
        pid_go = get_pid_go_by_proteinCode(path_dict['test_pid_go_file'], proteinCode=10090)
    elif mode == 'without_mouse':
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{specProteinTrain_path}/{dataset_name}-{hop}-hops-train-re-features-without-mouse.npz'))
            back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
            back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
        back_dict['train_y'] = ssp.load_npz(
            F'{specProteinTrain_path}/{dataset_name}_train_y_without_mouse.npz')
        pid_go = get_pid_go_by_proteinCode(path_dict['test_pid_go_file'], proteinCode=10090)

    back_dict['valid_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')
    back_dict['shap_idx'] = np.load(
        F'/data2/experiment_code/wangyansong/NAG_shap_data/DeepGraphGO-{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
        'arr_0']

    # test过程中需求的内容
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    mlb = get_mlb(path_dict['mlb_path'])
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])

    print(F'Protein Num in test: {len(pid_go)}')
    test_back_dict = {
        'test_pid_list': numpy.array(test_pid_list),
        'mlb': mlb,
        'pid_go': pid_go,
        'test_res_idx_': test_res_idx_
    }
    return back_dict, test_back_dict


# 加载CFAGO的数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
def load_protein_CFAGO_dataset_with_test(path_root, aspect, dataset_name, hops, top):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
        'test_y': None,
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features-top{top}.npz'))
        back_dict['valid_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features-top{top}.npz'))
        back_dict['test_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features-top{top}.npz'))
    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    return back_dict


# 加载CFAGO的数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 其中还加载了从CFAGO预训练/微调过程中保存下来的中间特征，这个特征分为预训练和微调两个阶段的
def load_protein_CFAGO_dataset_with_test_CFAGOmidfea(path_root, aspect, dataset_name, hops, top, midfeature):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
        'test_y': None,
    }
    if top == 300:
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features.npz'))
            back_dict['valid_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features.npz'))
            back_dict['test_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features.npz'))
    else:
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features-top{top}.npz'))
            back_dict['valid_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features-top{top}.npz'))
            back_dict['test_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features-top{top}.npz'))
    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    if midfeature == 'per':
        with open(
                F'/data2/experiment_code/wangyansong/NAG_CFAGO_dataset/CFAGO_data_midfeature/pretraining_{aspect}_{dataset_name}.pt',
                'rb') as f:
            midfeature_raw = pickle.load(f)
            print('Load pre-training mid-Feature from CFAGO')
    else:
        with open(
                F'/data2/experiment_code/wangyansong/NAG_CFAGO_dataset/CFAGO_data_midfeature/finetuning_{aspect}_{dataset_name}.pt',
                'rb') as f:
            midfeature_raw = pickle.load(f)
            print('Load fine-tuning mid-Feature from CFAGO')
    return back_dict, midfeature_raw


# 加载CFAGO的数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 其中还加载了从CFAGO预训练/微调过程中保存下来的中间特征，这个特征分为预训练和微调两个阶段的
# 还增添了过滤CFAGO InterPro特征的索引，但是索引是来自标准NAG模型，NAG concat CFAGO midFeature无法适应Shap的计算，只能用之前计算好的
def load_protein_CFAGO_dataset_with_test_CFAGOmidfea_filter(path_root, aspect, dataset_name, hops, top, midfeature,
                                                            t=0):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
        'test_y': None,
    }
    if top == 300:
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features.npz'))
            back_dict['valid_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features.npz'))
            back_dict['test_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features.npz'))
    else:
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features-top{top}.npz'))
            back_dict['valid_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features-top{top}.npz'))
            back_dict['test_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features-top{top}.npz'))
    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    back_dict['shap_idx'] = np.load(
        F'/data2/experiment_code/wangyansong/NAG_shap_data/{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
        'arr_0']
    print(F'Load shap values, t is {t}')
    if midfeature == 'per':
        with open(
                F'/data2/experiment_code/wangyansong/NAG_CFAGO_dataset/CFAGO_data_midfeature/pretraining_{aspect}_{dataset_name}.pt',
                'rb') as f:
            midfeature_raw = pickle.load(f)
            print('Load pre-training mid-Feature from CFAGO')
    else:
        with open(
                F'/data2/experiment_code/wangyansong/NAG_CFAGO_dataset/CFAGO_data_midfeature/finetuning_{aspect}_{dataset_name}.pt',
                'rb') as f:
            midfeature_raw = pickle.load(f)
            print('Load fine-tuning mid-Feature from CFAGO')
    return back_dict, midfeature_raw


# 加载CFAGO的数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
def load_protein_CFAGO_dataset_with_test_noTop(path_root, aspect, dataset_name, hops):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
        'test_y': None,
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features.npz'))
    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    return back_dict


# 加载CFAGO的数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值
def load_protein_CFAGO_dataset_with_test_noTop_filter(path_root, aspect, dataset_name, hops, t=0):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
        'test_y': None,
        'shap_idx': None
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features.npz'))
    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    back_dict['shap_idx'] = np.load(
        F'/data2/experiment_code/wangyansong/NAG_shap_data/{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
        'arr_0']
    return back_dict


# 加载CFAGO的数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用，特征是InterPro特征
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值，增加了top参数 使其可以加载不同top的数据
def load_protein_CFAGO_dataset_with_test_Top_filter(path_root, aspect, dataset_name, hops, top, t=0):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
        'test_y': None,
        'shap_idx': None
    }
    if top == 300:
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features.npz'))
            back_dict['valid_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features.npz'))
            back_dict['test_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features.npz'))
    else:
        for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
            back_dict['train_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features-top{top}.npz'))
            back_dict['valid_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features-top{top}.npz'))
            back_dict['test_x'].append(
                ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features-top{top}.npz'))
    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    back_dict['shap_idx'] = np.load(
        F'/data2/experiment_code/wangyansong/NAG_shap_data/{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
        'arr_0']
    return back_dict


# 加载CFAGO的数据集供双分支的NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 一个分支是InterPro特征，另一个分支是亚细胞位置
def load_branch_protein_CFAGO_dataset_with_test(path_root, aspect, dataset_name, hops):
    back_dict = {
        'train_x_interPro': [],
        'train_x_location': [],
        'train_y': None,
        'valid_x_interPro': [],
        'valid_x_location': [],
        'valid_y': None,
        'test_x_interPro': [],
        'test_x_location': [],
        'test_y': None,
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load interPro & location features'):
        back_dict['train_x_interPro'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x_interPro'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x_interPro'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features.npz'))

        back_dict['train_x_location'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-location-features.npz'))
        back_dict['valid_x_location'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-location-features.npz'))
        back_dict['test_x_location'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-location-features.npz'))
    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    return back_dict


# 加载CFAGO的数据集供双分支的NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 一个分支是InterPro特征，另一个分支是亚细胞位置
# 两个分支是不一样的，hops数也不一样
def load_branch_different_protein_CFAGO_dataset_with_test(path_root, aspect, dataset_name, hops_interPro,
                                                          hops_location):
    back_dict = {
        'train_x_interPro': [],
        'train_x_location': [],
        'train_y': None,
        'valid_x_interPro': [],
        'valid_x_location': [],
        'valid_y': None,
        'test_x_interPro': [],
        'test_x_location': [],
        'test_y': None,
    }
    for hop in tqdm(range(hops_interPro + 1), total=hops_interPro + 1, desc='load interPro features'):
        back_dict['train_x_interPro'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x_interPro'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x_interPro'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features.npz'))

    for hop in tqdm(range(hops_location + 1), total=hops_location + 1, desc='load location features'):
        back_dict['train_x_location'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-location-features.npz'))
        back_dict['valid_x_location'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-location-features.npz'))
        back_dict['test_x_location'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-location-features.npz'))

    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    return back_dict


# 加载CFAGO的数据集供单分支多特征输入的的NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 需要将InterPro特征和location两部分特征矩阵拼接在一起
def load_concat_protein_CFAGO_dataset_with_test(path_root, aspect, dataset_name, hops):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
        'test_y': None,
    }
    train_x_interPro = []
    train_x_location = []

    valid_x_interPro = []
    valid_x_location = []

    test_x_interPro = []
    test_x_location = []
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load interPro & location features'):
        train_x_interPro.append(ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-re-features.npz'))
        valid_x_interPro.append(ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-re-features.npz'))
        test_x_interPro.append(ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-re-features.npz'))

        train_x_location.append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-location-features.npz'))
        valid_x_location.append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-location-features.npz'))
        test_x_location.append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-location-features.npz'))

    for hop in tqdm(range(hops + 1), total=hops + 1, desc='concat interPro & location features'):
        back_dict['train_x'].append(ssp.hstack([train_x_interPro[hop], train_x_location[hop]], format='csr'))
        back_dict['valid_x'].append(ssp.hstack([valid_x_interPro[hop], valid_x_location[hop]], format='csr'))
        back_dict['test_x'].append(ssp.hstack([test_x_interPro[hop], test_x_location[hop]], format='csr'))

    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    return back_dict


# 加载CFAGO的数据集中的结构域+亚细胞位置数据供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
def load_CFAGO_dataset_only_with_test(path_root, aspect, dataset_name, hops, top):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
        'test_y': None,
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load CFAGO feature'):
        back_dict['train_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-CFAGO-features-top{top}.npz'))
        back_dict['valid_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-CFAGO-features-top{top}.npz'))
        back_dict['test_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-CFAGO-features-top{top}.npz'))
    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    return back_dict


# 加载CFAGO的数据集中的结构域+亚细胞位置数据供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值
def load_CFAGO_dataset_only_with_test_filter(path_root, aspect, dataset_name, hops, top, t=0):
    back_dict = {
        'train_x': [],
        'train_y': None,
        'valid_x': [],
        'valid_y': None,
        'test_x': [],
        'test_y': None,
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load CFAGO feature'):
        back_dict['train_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-train-CFAGO-features-top{top}.npz'))
        back_dict['valid_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-valid-CFAGO-features-top{top}.npz'))
        back_dict['test_x'].append(
            ssp.load_npz(F'{path_root}/{aspect}-{dataset_name}-{hop}-hops-test-CFAGO-features-top{top}.npz'))
    back_dict['train_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-train-y.npy')
    back_dict['valid_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-valid-y.npy')
    back_dict['test_y'] = np.load(F'{path_root}/{aspect}-{dataset_name}-test-y.npy')
    back_dict['shap_idx'] = np.load(
        F'/data2/experiment_code/wangyansong/NAG_shap_data/{dataset_name}-CFAGOonly_shap_values-filter-t{t}-onTest.npz')[
        'arr_0']
    return back_dict


# 加载test数据集供NAG训练好的模型推理测试使用 注意该方法仅加载test集供模型推理测试使用 目前已经弃用
def load_protein_dataset4model_test(path_root, dataset_name, hops):
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_pid_list_file': F'{path_root}/{dataset_name}_test_pid_list.txt',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_x = []
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        test_x.append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    pid_go = get_pid_go(path_dict['test_pid_go_file'])
    return test_x, test_pid_list, mlb, pid_go, test_res_idx_


# 加载test数据集供NAG训练好的模型推理测试使用
# 注意该方法仅加载test集供模型推理测试使用
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值
# 该方法支持使用蛋白质的物种code过滤出特定物种的样本
def load_protein_dataset4model_testBy_protein_code(path_root, dataset_name, hops, t):
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_pid_list_file': F'{path_root}/{dataset_name}_test_pid_list.txt',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }

    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_x = []
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        test_x.append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    if t is None:
        shap_idx = None
    else:
        shap_idx = np.load(
            F'/data2/experiment_code/wangyansong/NAG_shap_data/DeepGraphGO-{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
            'arr_0']
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    # 过滤人类
    pid_go_9606 = get_pid_go_by_proteinCode(path_dict['test_pid_go_file'], proteinCode=9606)
    # 过滤小鼠
    pid_go_10090 = get_pid_go_by_proteinCode(path_dict['test_pid_go_file'], proteinCode=10090)
    return test_x, shap_idx, test_pid_list, mlb, pid_go_9606, pid_go_10090, test_res_idx_


# 加载test数据集供NAG训练好的模型推理测试使用
# 注意该方法仅加载test集供模型推理测试使用
# 增加了使用shap工具计算后得出的应该提取的特征下标，t是过滤阈值
# 该方法支持分离在STRING网络中的蛋白质
def load_protein_dataset4model_testIN_STR(path_root, dataset_name, hops, t):
    # 路径list
    path_dict = {
        'feature_path': F'{path_root}/ppi_interpro.npz',
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_pid_list_file': F'{path_root}/{dataset_name}_test_pid_list.txt',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }

    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_x = []
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        test_x.append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    if t is None:
        shap_idx = None
    else:
        shap_idx = np.load(
            F'/data2/experiment_code/wangyansong/NAG_shap_data/DeepGraphGO-{dataset_name}-InterPro_shap_values-filter-t{t}-onTest.npz')[
            'arr_0']
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    # 过滤STR
    pid_go_STR = get_pid_go_by_STRHOMO(path_dict['test_pid_go_file'],
                                       [pid for pid in test_pid_list if pid in net_pid_map])
    # 过滤HOMO
    a, test_pid_list_only_HOMO, c = \
        get_homoNotSTR_ppi_idx(pid_list=test_pid_list, fasta_file=path_dict['test_fasta_file'],
                               net_pid_map=net_pid_map, net_blastdb=path_dict['net_blastdb_path'],
                               blast_output_path=path_dict['test_blast_output_path'])
    pid_go_HOMO = get_pid_go_by_STRHOMO(path_dict['test_pid_go_file'], test_pid_list_only_HOMO)

    return test_x, shap_idx, test_pid_list, mlb, pid_go_STR, pid_go_HOMO, test_res_idx_


# 加载数据集供NAG+GCN模型训练使用，注意 该方法不支持有test集训练的模型使用
def load_protein_dataset4GCN_NAG(path_root, dataset_name, hops):
    back_dict = {
        'train_x': [],
        'train_ppi': None,
        'train_y': None,
        'valid_x': [],
        'valid_ppi': None,
        'valid_y': None,
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
    back_dict['train_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-train-y.npz')
    back_dict['valid_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')
    # 用来对应样本和图节点的
    back_dict['train_ppi'] = np.load(F'{path_root}/{dataset_name}-train-ppi.npy')
    back_dict['valid_ppi'] = np.load(F'{path_root}/{dataset_name}-valid-ppi.npy')
    # dgl图加载
    dgl_graph = dgl.data.utils.load_graphs(F'{path_root}/ppi_dgl_top_100', )[0][0]
    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    self_loop[dgl_graph.edge_ids(nr_ := np.arange(dgl_graph.number_of_nodes()), nr_)] = 1.0
    dgl_graph.edata['self'] = self_loop
    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float().cuda()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float().cuda()
    # raw特征加载
    network_x = ssp.load_npz(F'{path_root}/ppi_interpro.npz')
    return back_dict, dgl_graph, network_x


# 加载数据集供NAG+GCN模型训练使用，注意 该方法支持有test集训练的模型使用
def load_protein_dataset4GCN_NAG_with_test(path_root, dataset_name, hops):
    back_dict = {
        'train_x': [],
        'train_ppi': None,
        'train_y': None,
        'valid_x': [],
        'valid_ppi': None,
        'valid_y': None,
        'test_x': [],
        'test_ppi': None,
    }
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        back_dict['train_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-train-re-features.npz'))
        back_dict['valid_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-valid-re-features.npz'))
        back_dict['test_x'].append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    back_dict['train_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-train-y.npz')
    back_dict['valid_y'] = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')
    # 用来对应样本和图节点的
    back_dict['train_ppi'] = np.load(F'{path_root}/{dataset_name}-train-ppi.npy')
    back_dict['valid_ppi'] = np.load(F'{path_root}/{dataset_name}-valid-ppi.npy')
    back_dict['test_ppi'] = np.load(F'{path_root}/{dataset_name}-test-ppi.npy')
    # dgl图加载
    dgl_graph = dgl.data.utils.load_graphs(F'{path_root}/ppi_dgl_top_100', )[0][0]
    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    self_loop[dgl_graph.edge_ids(nr_ := np.arange(dgl_graph.number_of_nodes()), nr_)] = 1.0
    dgl_graph.edata['self'] = self_loop
    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float().cuda()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float().cuda()
    # raw特征加载
    network_x = ssp.load_npz(F'{path_root}/ppi_interpro.npz')
    # test过程中需求的内容
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    pid_go = get_pid_go(path_dict['test_pid_go_file'])
    test_back_dict = {
        'test_pid_list': numpy.array(test_pid_list),
        'mlb': mlb,
        'pid_go': pid_go,
        'test_res_idx_': test_res_idx_
    }
    return back_dict, dgl_graph, network_x, test_back_dict


# 加载test数据集供NAG+GCN训练好的模型推理测试使用 注意该方法仅加载test集供模型推理测试使用 目前已经弃用
def load_protein_dataset4GCN_NAG_model_test(path_root, dataset_name, hops):
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'test_pid_list_file': F'{path_root}/{dataset_name}_test_pid_list.txt',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'test_pid_go_file': F'{path_root}/{dataset_name}_test_go.txt',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    test_x = []
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load re feature'):
        test_x.append(ssp.load_npz(F'{path_root}/{dataset_name}-{hop}-hops-test-re-features.npz'))
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    mlb = get_mlb(path_dict['mlb_path'])
    pid_go = get_pid_go(path_dict['test_pid_go_file'])

    # dgl图加载
    dgl_graph = dgl.data.utils.load_graphs(F'{path_root}/ppi_dgl_top_100', )[0][0]
    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    self_loop[dgl_graph.edge_ids(nr_ := np.arange(dgl_graph.number_of_nodes()), nr_)] = 1.0
    dgl_graph.edata['self'] = self_loop
    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float().cuda()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float().cuda()
    # raw特征加载
    network_x = ssp.load_npz(F'{path_root}/ppi_interpro.npz')
    return test_x, numpy.array(test_pid_list), mlb, pid_go, test_res_idx_, test_ppi, dgl_graph, network_x


# 从1-hops特征矩阵开始计算到hops指定的，保存为pt格式的张量，目前已弃用
def build_re_features(adj_path, path_root, hops, lpe_flag=False, pe_dim=3):
    # 路径list
    path_dict = {
        'feature_path': path_root + '/ppi_interpro.npz',
    }
    # 加载构建完成的sparse_coo张量邻接矩阵
    adj = torch.load(adj_path)
    # 构建特征张量 也是sparse_coo
    features_csr = ssp.load_npz(path_dict['feature_path'])
    features_coo = ssp.coo_matrix(features_csr)
    features = torch.sparse_coo_tensor(np.array([features_coo.row, features_coo.col]),
                                       features_coo.data,
                                       size=torch.Size(features_coo.shape),
                                       dtype=torch.float32)
    # 是否添加拉普拉斯特征向量
    if lpe_flag:
        src, dst = adj._indices()
        graph = dgl.graph((src.numpy(), dst.numpy()), num_nodes=adj.shape[0])
        lpe = utils.laplacian_positional_encoding(graph, pe_dim)
        features = torch.cat((features, lpe), dim=1)

    # 按照NAG的方式收集邻域信息
    x = features
    for i in tqdm(range(1, hops + 1), total=hops, desc="matrix multiple in hops " + str(hops)):
        x = torch.sparse.mm(adj, x)
        torch.save(x, path_root + '/' + str(i) + '-hops-re-features.pt')


# 从old_hops指定的hops特征矩阵开始计算到hops指定的，保存为pt格式的张量，目前已弃用
def build_re_features_from_saved(adj_path, path_root, old_hops, hops):
    # 路径list
    path_dict = {
        'feature_path': path_root + '/ppi_interpro.npz',
        'saved_features': path_root + '/' + str(old_hops) + '-hops-re-features.pt',
    }
    # 加载构建完成的sparse_coo张量邻接矩阵
    adj = torch.load(adj_path)
    features = torch.load(path_dict['saved_features'])

    # 按照NAG的方式收集邻域信息
    x = features
    for i in tqdm(range(old_hops + 1, hops + 1), total=hops - old_hops,
                  desc="matrix multiple form hops " + str(old_hops) + " to " + str(hops)):
        x = torch.sparse.mm(adj, x)
        torch.save(x, path_root + '/' + str(i) + '-hops-re-features.pt')


# 从1-hops特征矩阵开始计算到hops指定的，保存为csr格式的稀疏矩阵
def build_re_features_csr(adj_path, path_root, hops):
    # 路径list
    path_dict = {
        'feature_path': path_root + '/ppi_interpro.npz',
    }
    # 加载构建完成的sparse_coo张量邻接矩阵
    adj = ssp.load_npz(adj_path)
    # 构建特征
    features_csr = ssp.load_npz(path_dict['feature_path'])

    # 按照NAG的方式收集邻域信息
    x = features_csr
    for i in tqdm(range(1, hops + 1), total=hops, desc="matrix multiple in hops " + str(hops)):
        x = adj.dot(x)
        ssp.save_npz(path_root + '/' + str(i) + '-hops-re-features.npz', x.astype('float32'))


# 从old_hops指定的hops特征矩阵开始计算到hops指定的，保存为csr格式的稀疏矩阵
def build_re_features_from_saved_csr(adj_path, path_root, old_hops, hops):
    # 路径list
    path_dict = {
        'saved_features': path_root + '/' + str(old_hops) + '-hops-re-features.npz',
    }
    # 加载构建完成的sparse_coo张量邻接矩阵
    adj = ssp.load_npz(adj_path)
    features_csr = ssp.load_npz(path_dict['saved_features'])

    # 按照NAG的方式收集邻域信息
    x = features_csr
    for i in tqdm(range(old_hops + 1, hops + 1), total=hops - old_hops,
                  desc="matrix multiple form hops " + str(old_hops) + " to " + str(hops)):
        x = adj.dot(x)
        ssp.save_npz(path_root + '/' + str(i) + '-hops-re-features.npz', x.astype('float32'))


# 单独保存了样本对应的PPI图中的节点编号列表，为了输入NAG_GCN模型
def save_ppi_list(path_root, dataset_name):
    # 路径list
    path_dict = {
        'net_pid_list_path': F'{path_root}/ppi_pid_list.txt',
        'net_blastdb_path': F'{path_root}/ppi_blastdb',
        'train_pid_list_file': F'{path_root}/{dataset_name}_train_pid_list.txt',
        'train_fasta_file': F'{path_root}/{dataset_name}_train.fasta',
        'train_pid_go_file': F'{path_root}/{dataset_name}_train_go.txt',
        'valid_pid_list_file': F'{path_root}/{dataset_name}_valid_pid_list.txt',
        'valid_fasta_file': F'{path_root}/{dataset_name}_valid.fasta',
        'valid_pid_go_file': F'{path_root}/{dataset_name}_valid_go.txt',
        'test_pid_list_file': F'{path_root}/{dataset_name}_test_pid_list.txt',
        'test_fasta_file': F'{path_root}/{dataset_name}_test.fasta',
        'mlb_path': Path(F'{path_root}/{dataset_name}_go.mlb'),
        'valid_blast_output_path': Path(F'{path_root}/{dataset_name}-valid-ppi-blast-out'),
        'test_blast_output_path': Path(F'{path_root}/{dataset_name}-test-ppi-blast-out'),
    }
    # 加载网络的pid列表和字典
    net_pid_list = get_pid_list(path_dict['net_pid_list_path'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}

    # 加载数据集中的pid列表及其GO标签列表
    train_pid_list, _, train_go = get_data(fasta_file=path_dict['train_fasta_file'],
                                           pid_go_file=path_dict['train_pid_go_file'])
    valid_pid_list, _, valid_go = get_data(fasta_file=path_dict['valid_fasta_file'],
                                           pid_go_file=path_dict['valid_pid_go_file'])
    test_pid_list, _, test_go = get_data(fasta_file=path_dict['test_fasta_file'])

    # 使用MLB映射GO标签
    mlb = get_mlb(path_dict['mlb_path'], train_go)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_y, valid_y = mlb.transform(train_go).astype(np.float32), mlb.transform(valid_go).astype(np.float32)
    # 将各集合样本与图中节点对应 并给出对应的标签y(mlb映射后6640维的)
    *_, train_ppi, train_y = get_ppi_idx(train_pid_list, train_y, net_pid_map)
    *_, valid_ppi, valid_y = get_homo_ppi_idx(valid_pid_list, path_dict['valid_fasta_file'],
                                              valid_y, net_pid_map, path_dict['net_blastdb_path'],
                                              path_dict['valid_blast_output_path'])
    # test集合是没有go标签的
    test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, path_dict['test_fasta_file'],
                                                                  None, net_pid_map, path_dict['net_blastdb_path'],
                                                                  path_dict['test_blast_output_path'])
    np.save(F'{path_root}/{dataset_name}-train-ppi.npy', np.array(train_ppi))
    np.save(F'{path_root}/{dataset_name}-valid-ppi.npy', np.array(valid_ppi))
    np.save(F'{path_root}/{dataset_name}-test-ppi.npy', np.array(test_ppi))


# 加载数据集供NAG模型训练使用，注意 该方法支持有test集训练的模型使用
def summary_DeepGraphGO_dataset(path_root, dataset_name):
    import pandas as pd
    train_y = ssp.load_npz(F'{path_root}/{dataset_name}-train-y.npz')
    valid_y = ssp.load_npz(F'{path_root}/{dataset_name}-valid-y.npz')

    train_y_dense = train_y.toarray().astype(int)
    train_y_sum = np.sum(train_y_dense, axis=0)

    valid_y_dense = valid_y.toarray().astype(int)
    valid_y_sum = np.sum(valid_y_dense, axis=0)

    # test过程中需求的内容
    # pid_go = get_pid_go(F'{path_root}/{dataset_name}_test_go.txt')

    # 将NumPy数组转换为pandas的DataFrame
    df = pd.DataFrame({'train_y_sum': train_y_sum, 'valid_y_sum': valid_y_sum})

    # 将DataFrame写入Excel文件
    excel_file_path = 'output_file.xlsx'
    df.to_excel(excel_file_path, index=False)

    print(111)


if __name__ == '__main__':
    # get_protein_dataset(path_root='dataset', dataset_name='cc', hops=9)
    # get_protein_dataset_from_hops(path_root='dataset', dataset_name='mf', start_hop=9, end_hop=9)
    # build_re_features(adj_path='dataset/ppi_adj_top_100_norm.pt', path_root='dataset', hops=2)
    # build_re_features_from_saved(adj_path='dataset/ppi_adj_top_100_norm.pt', path_root='dataset', old_hops=2, hops=3)
    # a = ssp.load_npz('/data1/experiment_code/wangyansong/NAGphormer-main/dataset/2-hops-re-features.npz')
    # csr_matrix = utils.torch_sparse_tensor_to_sparse_mx(a).tocsr()
    # ssp.save_npz('dataset/2-hops-re-features.npz', csr_matrix)
    # build_re_features_csr(adj_path='dataset/ppi_adj_top_100_norm.npz', path_root='dataset', hops=10)
    # build_re_features_from_saved_csr(adj_path='dataset/ppi_adj_top_100_norm.npz', path_root='dataset', old_hops=4,
    #                                  hops=9)
    # save_ppi_list('dataset', 'cc')
    # q = np.load('dataset/mf-train-ppi.npy', allow_pickle=True)
    # print(11)
    # load_concat_protein_CFAGO_dataset_with_test(path_root='dataset_CFAGO', aspect='human', dataset_name='mf', hops=2)
    # summary_DeepGraphGO_dataset(path_root='dataset', dataset_name='bp')
    # build_laplacian_code()
    # loaded_tensor = torch.load('/data2/experiment_code/wangyansong/tensor.pt')
    # pid_go_human = get_pid_go_by_proteinCode('dataset/mf_test_go.txt', proteinCode=9606)
    # pid_go_mouse = get_pid_go_by_proteinCode('dataset/mf_test_go.txt', proteinCode=10090)
    mlb = get_mlb(Path('/data1/experiment_code/wangyansong/NAGphormer-main/dataset/mf_go.mlb'))
    a = list(mlb.classes_)
    for i in ['GO:0016773', 'GO:0016301', 'GO:0016772', 'GO:0140096', 'GO:0016740', 'GO:0003824']:
        print(a.index(i))
    print(111)
