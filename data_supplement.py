import os
import numpy as np
import warnings
import scipy.sparse as ssp
import joblib
import torch
import dgl.data
from logzero import logger
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer

from data import get_pid_list, get_mlb
from utils import psiblast


def get_ppi_idx(pid_list, net_pid_map):
    pid_list_ = []
    for i, pid in enumerate(pid_list):
        pid_list_.append((i, pid, net_pid_map[pid]))
    pid_list_ = tuple(zip(*pid_list_))
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2]


def get_homo_ppi_idx(pid_list, fasta_file, net_pid_map, net_blastdb, blast_output_path):
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


def get_STRI(domain):
    """
    proteins in STRING
    """

    net_pid_list = get_pid_list(pid_list_file='dataset/ppi_pid_list.txt')
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}

    test_pid_list = get_pid_list(F'dataset/{domain}_test_pid_list.txt')
    # protein in STRING
    test_pid_list = [pid for pid in test_pid_list if pid in net_pid_map]
    # STRI 蛋白质在测试集中的 index、STRI 蛋白质 id 以及蛋白质在 ppi 中的 index
    test_res_idx_, test_pid_list_, test_ppi = get_ppi_idx(pid_list=test_pid_list, net_pid_map=net_pid_map)

    test_mlb = get_mlb(Path(F'dataset/{domain}_go.mlb'))
    test_labels_num = len(test_mlb.classes_)

    logger.info(F'Number of Test Set Labels: {test_labels_num}')
    logger.info(F'Size of Test Set: {len(test_ppi)}\n')

    return test_ppi, test_pid_list, test_mlb, test_labels_num, test_res_idx_


def get_HOMO(domain):
    """
    proteins being not in STRING but homologous to proteins in STRING
    """

    net_pid_list = get_pid_list(pid_list_file='dataset/ppi_pid_list.txt')
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}

    test_pid_list = get_pid_list(F'dataset/{domain}_test_pid_list.txt')
    # proteins being not in STRING but homologous to proteins in STRING
    net_blastdb = F'dataset/ppi_blastdb'
    test_fasta_file = F'dataset/{domain}_test.fasta'
    test_blast_output_path = F'dataset/{domain}-test-ppi-blast-out'
    # HOMO 蛋白质在测试集中的 index、HOMO 蛋白质 id 以及蛋白质在 ppi 中的 index
    test_res_idx_, test_pid_list_, test_ppi = \
        get_homo_ppi_idx(pid_list=test_pid_list, fasta_file=test_fasta_file,
                         net_pid_map=net_pid_map, net_blastdb=net_blastdb, blast_output_path=test_blast_output_path)

    test_mlb = get_mlb(Path(F'data/{domain}_go.mlb'))
    test_labels_num = len(test_mlb.classes_)

    logger.info(F'Number of Test Set Labels: {test_labels_num}')
    logger.info(F'Size of Test Set: {len(test_ppi)}\n')

    return test_ppi, test_pid_list, test_mlb, test_labels_num, test_res_idx_


def get_HUMAN():
    """
    proteins in HUMAN
    """


def get_MOUSE():
    """
    proteins in MOUSE
    """


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # q = get_STRI('mf')
    q = get_HOMO('mf')
    get_STRI('bp')
    get_HOMO('bp')
    get_STRI('cc')
    get_HOMO('cc')
