import pickle
import numpy as np
import scipy.io as sio
import torch
from sklearn.preprocessing import minmax_scale
import scipy.sparse as ssp
from tqdm import trange, tqdm


# 对ppi网络归一化使用的工具，输入是一个稀疏矩阵，csr csc格式都行
def get_norm_net_mat(net_mat):
    degree_0 = np.asarray(net_mat.sum(0)).squeeze()
    mat_d_0 = ssp.diags(degree_0 ** -0.5, format='csr')
    degree_1 = np.asarray(net_mat.sum(1)).squeeze()
    mat_d_1 = ssp.diags(degree_1 ** -0.5, format='csr')
    return mat_d_0 @ net_mat @ mat_d_1


# 处理CFAGO数据集中的PPI网络，需要经过minmax放缩、加自反边、归一化，最后存成csr格式的稀疏矩阵
def process_ppi_net(path, output_path, top):
    PPI_raw = sio.loadmat(path, squeeze_me=True)
    PPI_net = PPI_raw['Net']
    PPI_net_dense = PPI_net.todense()
    PPI_net_dense_scale = minmax_scale(PPI_net_dense)
    PPI_net_csr_scale_self = ssp.csr_matrix(PPI_net_dense_scale) + ssp.eye(PPI_net_dense_scale.shape[0], format='csr')
    resource, destination, values = [], [], []
    for i in trange(PPI_net_csr_scale_self.shape[0]):
        for v_, d_ in sorted(zip(PPI_net_csr_scale_self[i].data, PPI_net_csr_scale_self[i].indices), reverse=True)[
                      :top]:
            resource.append(i)
            destination.append(d_)
            values.append(v_)
    ppi_net_norm = get_norm_net_mat(
        ssp.csc_matrix((values, (resource, destination)), shape=PPI_net_csr_scale_self.shape).T)
    ssp.save_npz(output_path, ppi_net_norm)


# 按照NAG的方式处理InterPro特征矩阵
def build_re_features_csr(path_root, top, hops, aspect='human'):
    # 路径list
    path_dict = {
        'feature_path': path_root + F'/{aspect}_interpro_feature.npz',
        'adj_path': path_root + F'/{aspect}_ppi_adj_norm_top{top}.npz',
    }
    # 加载邻接矩阵  csr格式的
    adj = ssp.load_npz(path_dict['adj_path'])
    # 加载特征矩阵 csr格式的
    features_csr = ssp.load_npz(path_dict['feature_path'])

    # 按照NAG的方式收集邻域信息
    x = features_csr
    for i in tqdm(range(1, hops + 1), total=hops, desc="matrix multiple in hops " + str(hops)):
        x = adj.dot(x)
        ssp.save_npz(path_root + F'/{aspect}-{str(i)}-hops-re-features-top{top}.npz', x.astype('float32'))


# 按照NAG的方式处理亚细胞位置和结构域特征矩阵
# 注意，原始特征文件中包含亚细胞位置和结构域特征两部分，需要拆分开
def build_location_features_csr(path_root, top, hops, aspect='human'):
    # 路径list
    path_dict = {
        'feature_path': path_root + F'/{aspect}_location_feature.npz',
        'adj_path': path_root + F'/{aspect}_ppi_adj_norm_top{top}.npz',
    }
    # 加载邻接矩阵  csr格式的
    adj = ssp.load_npz(path_dict['adj_path'])
    # 加载特征矩阵 csr格式的
    features_csr = ssp.load_npz(path_dict['feature_path'])
    # 拆分出亚细胞位置特征
    location_features_csr = features_csr[:, :442]

    # 按照NAG的方式收集邻域信息
    x = location_features_csr
    for i in tqdm(range(1, hops + 1), total=hops, desc="matrix multiple in hops " + str(hops)):
        x = adj.dot(x)
        ssp.save_npz(path_root + F'/{aspect}-{str(i)}-hops-location-features-top{top}.npz', x.astype('float32'))


# 按照NAG的方式处理亚细胞位置和结构域特征矩阵
# 注意，原始特征文件中包含亚细胞位置和结构域特征两部分，此处我们不做拆分，将这两部分特征全部使用
def build_CFAGO_features_csr(path_root, top, hops, aspect='human'):
    # 路径list
    path_dict = {
        'feature_path': path_root + F'/{aspect}_location_feature.npz',
        'adj_path': path_root + F'/{aspect}_ppi_adj_norm_top{top}.npz',
    }
    # 加载邻接矩阵  csr格式的
    adj = ssp.load_npz(path_dict['adj_path'])
    # 加载特征矩阵 csr格式的
    features_csr = ssp.load_npz(path_dict['feature_path'])

    # 按照NAG的方式收集邻域信息
    x = features_csr
    for i in tqdm(range(1, hops + 1), total=hops, desc="matrix multiple in hops " + str(hops)):
        x = adj.dot(x)
        ssp.save_npz(path_root + F'/{aspect}-{str(i)}-hops-CFAGO-features-top{top}.npz', x.astype('float32'))


# 读取蛋白质注释字典中的数据 并将他们按照bpo mfo cco + train valid test的划分方式区独立存储
def process_annotation(path_root, aspect='human'):
    # 路径list
    path_dict = {
        'annotation_path': path_root + F'/{aspect}_annot.mat',
    }

    # ===========load annot============
    Annot = sio.loadmat(path_dict['annotation_path'], squeeze_me=True)

    # 这块是用来提取下标的，这个没必要另外保存，直接读取了用来抽取特征矩阵的行就行，写在这备用
    # idx = Annot['indx'].tolist()
    # idx_bpo = idx[0].tolist()
    # idx_mfo = idx[1].tolist()
    # idx_cco = idx[2].tolist()

    # 处理GO标签 这个标签是二值化之后的，例如human BPO是45分类，则每个样本是一个45维度的二值化向量
    GO = Annot['GO'].tolist()
    # 处理BPO
    GO_bpo = GO[0].tolist()
    np.save(path_root + F'/{aspect}_bp_train_y', GO_bpo[0])
    np.save(path_root + F'/{aspect}_bp_valid_y', GO_bpo[1])
    np.save(path_root + F'/{aspect}_bp_test_y', GO_bpo[2])
    # 处理MFO
    GO_mfo = GO[1].tolist()
    np.save(path_root + F'/{aspect}_mf_train_y', GO_mfo[0])
    np.save(path_root + F'/{aspect}_mf_valid_y', GO_mfo[1])
    np.save(path_root + F'/{aspect}_mf_test_y', GO_mfo[2])
    # 处理CCO
    GO_cco = GO[2].tolist()
    np.save(path_root + F'/{aspect}_cc_train_y', GO_cco[0])
    np.save(path_root + F'/{aspect}_cc_valid_y', GO_cco[1])
    np.save(path_root + F'/{aspect}_cc_test_y', GO_cco[2])


# 按照注释字典中标注的下标，从做完NAG聚合操作的InterPro特征矩阵中抽取需要的行，组成BPO MFO CCO 的训练 验证 测试 集
def process_features_csr4model_train(path_root='dataset_CFAGO', hops=9, top=300, aspect='human'):
    # 路径list
    path_dict = {
        'annotation_path': path_root + F'/{aspect}_annot.mat',
        'feature_path': path_root + F'/{aspect}_interpro_feature.npz',
    }

    # ===========load annot============
    Annot = sio.loadmat(path_dict['annotation_path'], squeeze_me=True)

    # 读取下标
    idx = Annot['indx'].tolist()
    idx_bpo = idx[0].tolist()
    idx_mfo = idx[1].tolist()
    idx_cco = idx[2].tolist()

    # 加载特征并提取特征
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load, extract and save interPro feature:'):
        if hop == 0:
            features_csr = ssp.load_npz(path_dict['feature_path'])
        else:
            features_csr = ssp.load_npz(F'{path_root}/{aspect}-{str(hop)}-hops-re-features-top{top}.npz')
        # 提取bpo特征
        ssp.save_npz(F'{path_root}/{aspect}-bp-{hop}-hops-train-re-features-top{top}.npz', features_csr[idx_bpo[0]])
        ssp.save_npz(F'{path_root}/{aspect}-bp-{hop}-hops-valid-re-features-top{top}.npz', features_csr[idx_bpo[1]])
        ssp.save_npz(F'{path_root}/{aspect}-bp-{hop}-hops-test-re-features-top{top}.npz', features_csr[idx_bpo[2]])
        # 提取mfo特征
        ssp.save_npz(F'{path_root}/{aspect}-mf-{hop}-hops-train-re-features-top{top}.npz', features_csr[idx_mfo[0]])
        ssp.save_npz(F'{path_root}/{aspect}-mf-{hop}-hops-valid-re-features-top{top}.npz', features_csr[idx_mfo[1]])
        ssp.save_npz(F'{path_root}/{aspect}-mf-{hop}-hops-test-re-features-top{top}.npz', features_csr[idx_mfo[2]])
        # 提取cco特征
        ssp.save_npz(F'{path_root}/{aspect}-cc-{hop}-hops-train-re-features-top{top}.npz', features_csr[idx_cco[0]])
        ssp.save_npz(F'{path_root}/{aspect}-cc-{hop}-hops-valid-re-features-top{top}.npz', features_csr[idx_cco[1]])
        ssp.save_npz(F'{path_root}/{aspect}-cc-{hop}-hops-test-re-features-top{top}.npz', features_csr[idx_cco[2]])


# 按照注释字典中标注的下标，从做完NAG聚合操作的亚细胞位置特征矩阵中抽取需要的行，组成BPO MFO CCO 的训练 验证 测试 集
def process_location_features_csr4model_train(path_root='dataset_CFAGO', hops=9, top=300, aspect='human'):
    # 路径list
    path_dict = {
        'annotation_path': path_root + F'/{aspect}_annot.mat',
        'feature_path': path_root + F'/{aspect}_location_feature.npz',
    }

    # ===========load annot============
    Annot = sio.loadmat(path_dict['annotation_path'], squeeze_me=True)

    # 读取下标
    idx = Annot['indx'].tolist()
    idx_bpo = idx[0].tolist()
    idx_mfo = idx[1].tolist()
    idx_cco = idx[2].tolist()

    # 加载特征并提取特征
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load, extract and save location feature:'):
        if hop == 0:
            features_csr_raw = ssp.load_npz(path_dict['feature_path'])
            # 拆分出亚细胞位置特征
            features_csr = features_csr_raw[:, :442]
        else:
            features_csr = ssp.load_npz(F'{path_root}/{aspect}-{str(hop)}-hops-location-features-top{top}.npz')
        # 提取bpo特征
        ssp.save_npz(F'{path_root}/{aspect}-bp-{hop}-hops-train-location-features-top{top}.npz',
                     features_csr[idx_bpo[0]])
        ssp.save_npz(F'{path_root}/{aspect}-bp-{hop}-hops-valid-location-features-top{top}.npz',
                     features_csr[idx_bpo[1]])
        ssp.save_npz(F'{path_root}/{aspect}-bp-{hop}-hops-test-location-features-top{top}.npz',
                     features_csr[idx_bpo[2]])

        # 提取mfo特征
        ssp.save_npz(F'{path_root}/{aspect}-mf-{hop}-hops-train-location-features-top{top}.npz',
                     features_csr[idx_mfo[0]])
        ssp.save_npz(F'{path_root}/{aspect}-mf-{hop}-hops-valid-location-features-top{top}.npz',
                     features_csr[idx_mfo[1]])
        ssp.save_npz(F'{path_root}/{aspect}-mf-{hop}-hops-test-location-features-top{top}.npz',
                     features_csr[idx_mfo[2]])

        # 提取cco特征
        ssp.save_npz(F'{path_root}/{aspect}-cc-{hop}-hops-train-location-features-top{top}.npz',
                     features_csr[idx_cco[0]])
        ssp.save_npz(F'{path_root}/{aspect}-cc-{hop}-hops-valid-location-features-top{top}.npz',
                     features_csr[idx_cco[1]])
        ssp.save_npz(F'{path_root}/{aspect}-cc-{hop}-hops-test-location-features-top{top}.npz',
                     features_csr[idx_cco[2]])


# 按照注释字典中标注的下标，从做完NAG聚合操作的CFAGO特征矩阵中抽取需要的行，组成BPO MFO CCO 的训练 验证 测试 集
def process_CFAGO_features_csr4model_train(path_root, hops=9, top=300, aspect='human'):
    # 路径list
    path_dict = {
        'annotation_path': path_root + F'/{aspect}_annot.mat',
        'feature_path': path_root + F'/{aspect}_location_feature.npz',
    }

    # ===========load annot============
    Annot = sio.loadmat(path_dict['annotation_path'], squeeze_me=True)

    # 读取下标
    idx = Annot['indx'].tolist()
    idx_bpo = idx[0].tolist()
    idx_mfo = idx[1].tolist()
    idx_cco = idx[2].tolist()

    # 加载特征并提取特征
    for hop in tqdm(range(hops + 1), total=hops + 1, desc='load, extract and save location feature:'):
        if hop == 0:
            features_csr = ssp.load_npz(path_dict['feature_path'])
        else:
            features_csr = ssp.load_npz(F'{path_root}/{aspect}-{str(hop)}-hops-CFAGO-features-top{top}.npz')
        # 提取bpo特征
        ssp.save_npz(F'{path_root}/{aspect}-bp-{hop}-hops-train-CFAGO-features-top{top}.npz',
                     features_csr[idx_bpo[0]])
        ssp.save_npz(F'{path_root}/{aspect}-bp-{hop}-hops-valid-CFAGO-features-top{top}.npz',
                     features_csr[idx_bpo[1]])
        ssp.save_npz(F'{path_root}/{aspect}-bp-{hop}-hops-test-CFAGO-features-top{top}.npz',
                     features_csr[idx_bpo[2]])

        # 提取mfo特征
        ssp.save_npz(F'{path_root}/{aspect}-mf-{hop}-hops-train-CFAGO-features-top{top}.npz',
                     features_csr[idx_mfo[0]])
        ssp.save_npz(F'{path_root}/{aspect}-mf-{hop}-hops-valid-CFAGO-features-top{top}.npz',
                     features_csr[idx_mfo[1]])
        ssp.save_npz(F'{path_root}/{aspect}-mf-{hop}-hops-test-CFAGO-features-top{top}.npz',
                     features_csr[idx_mfo[2]])

        # 提取cco特征
        ssp.save_npz(F'{path_root}/{aspect}-cc-{hop}-hops-train-CFAGO-features-top{top}.npz',
                     features_csr[idx_cco[0]])
        ssp.save_npz(F'{path_root}/{aspect}-cc-{hop}-hops-valid-CFAGO-features-top{top}.npz',
                     features_csr[idx_cco[1]])
        ssp.save_npz(F'{path_root}/{aspect}-cc-{hop}-hops-test-CFAGO-features-top{top}.npz',
                     features_csr[idx_cco[2]])


def summary_annotation(path_root, aspect='human'):
    # 路径list
    path_dict = {
        'annotation_path': path_root + F'/{aspect}_annot.mat',
    }

    # ===========load annot============
    Annot = sio.loadmat(path_dict['annotation_path'], squeeze_me=True)

    # 处理GO标签 这个标签是二值化之后的，例如human BPO是45分类，则每个样本是一个45维度的二值化向量
    GO = Annot['GO'].tolist()
    # 处理BPO
    GO_bpo = GO[0].tolist()
    GO_bpo_train = GO_bpo[0]
    GO_bpo_valid = GO_bpo[1]
    GO_bpo_test = GO_bpo[2]

    GO_bpo_train_sum = np.sum(GO_bpo_train, axis=0)
    GO_bpo_valid_sum = np.sum(GO_bpo_valid, axis=0)
    GO_bpo_test_sum = np.sum(GO_bpo_test, axis=0)

    # 处理MFO
    GO_mfo = GO[1].tolist()
    GO_mfo_train = GO_mfo[0]
    GO_mfo_valid = GO_mfo[1]
    GO_mfo_test = GO_mfo[2]

    GO_mfo_train_sum = np.sum(GO_mfo_train, axis=0)
    GO_mfo_valid_sum = np.sum(GO_mfo_valid, axis=0)
    GO_mfo_test_sum = np.sum(GO_mfo_test, axis=0)

    # 处理CCO
    GO_cco = GO[2].tolist()
    GO_cco_train = GO_cco[0]
    GO_cco_valid = GO_cco[1]
    GO_cco_test = GO_cco[2]

    GO_cco_train_sum = np.sum(GO_cco_train, axis=0)
    GO_cco_valid_sum = np.sum(GO_cco_valid, axis=0)
    GO_cco_test_sum = np.sum(GO_cco_test, axis=0)
    print(111)


if __name__ == '__main__':
    # a = ssp.load_npz('dataset/bp-train-y.npz')
    # a = ssp.load_npz('dataset_CFAGO/human_ppi_adj_norm.npz')
    # a = np.load('dataset_CFAGO/human_cc_train_y.npy')
    # print(111)
    # process_ppi_net(path='/data2/experiment_code/wangyansong/NAG_CFAGO_dataset/human_net_combined.mat',
    #                 output_path='/data2/experiment_code/wangyansong/NAG_CFAGO_dataset/human_ppi_adj_norm_top700.npz',
    #                 top=700)
    build_re_features_csr(path_root='dataset_CFAGO', hops=9, top=500)
    # process_annotation(path_root='dataset_CFAGO')
    # process_features_csr4model_train(top=500)
    # build_location_features_csr(path_root='dataset_CFAGO', hops=9, top=500)
    # process_location_features_csr4model_train(top=500)
    # build_CFAGO_features_csr(path_root='/data2/experiment_code/wangyansong/NAG_CFAGO_dataset', top=700, hops=9)
    # process_CFAGO_features_csr4model_train(path_root='/data2/experiment_code/wangyansong/NAG_CFAGO_dataset', top=700)
    # summary_annotation(path_root='/data2/experiment_code/wangyansong/NAG_CFAGO_dataset')
