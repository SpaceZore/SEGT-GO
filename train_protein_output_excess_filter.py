import os
import datetime
from itertools import chain
from thop import profile
import wandb
from tqdm import tqdm
import scipy.sparse as ssp
from sklearn.metrics import average_precision_score as aupr
from data import load_protein_dataset, get_pid_go_mat, get_pid_go_sc_mat, output_format, load_protein_dataset_with_test, \
    load_protein_dataset_with_test_filter
import time
import utils
import random
import numpy as np
import torch
import torch.nn as nn
from early_stop import EarlyStopping, Stop_args
from model import TransformerModel
from lr import PolynomialDecayLR
import torch.utils.data as Data
import argparse

from model_output_excess import TransformerModelOutputExcess

ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}


def get_time_stamp13():
    datetime_now = datetime.datetime.now()
    date_stamp = str(int(time.mktime(datetime_now.timetuple())))
    data_microsecond = str("%06d" % datetime_now.microsecond)[0:3]
    date_stamp = date_stamp + data_microsecond
    return date_stamp


# Training settings
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser()

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='mf',
                        help='Choose from {pubmed}')
    parser.add_argument('--root_path', type=str, default='dataset',
                        help='Dataset path.')
    parser.add_argument('--device', type=int, default=0,
                        help='Device cuda id')
    parser.add_argument('--t', type=float, default=0,
                        help='')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed.')

    # model parameters
    parser.add_argument('--hops', type=int, default=4,
                        help='Hop of neighbors to be calculated')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--att_dropout', type=float, default=0.1,
                        help='Dropout')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--tot_updates', type=int, default=1000,
                        help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=30,
                        help='Patience for early stopping')

    return parser.parse_args()


def train_valid_epoch(data, test_data, epoch, model, args, train_data_loader, val_data_loader,
                      test_data_loader, fn_loss, device, optimizer, lr_scheduler: PolynomialDecayLR):
    model.train()
    loss_train_b = 0

    total_train_iter = (data['train_y'].shape[0] + args.batch_size - 1) // args.batch_size
    for _, item in tqdm(enumerate(train_data_loader), total=total_train_iter,
                        desc=F'Epoch {epoch}'):

        batch_x = process_batch_x(data['train_x'], item, data['shap_idx'], model).to(device)
        batch_y = process_batch_y(data['train_y'], item).to(device)
        optimizer.zero_grad()

        flops, params = profile(model, inputs=(batch_x))
        print(flops / 1e6, params / 1e6)

        output = model(batch_x)
        loss_train = fn_loss(output, batch_y)
        loss_train.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_train_b += loss_train.item()

    model.eval()
    loss_val = 0
    fmax_val = 0
    aupr_val = 0
    with torch.no_grad():
        for _, item in enumerate(val_data_loader):
            batch_x = process_batch_x(data['valid_x'], item, data['shap_idx'], model).to(device)
            batch_y, y_csr = process_valid_batch_y(data['valid_y'], item)

            output = model(batch_x)
            loss_val += fn_loss(output, batch_y.to(device)).item()

            output_np = torch.sigmoid(output).cpu().numpy()
            (fmax_, t_) = utils.fmax(y_csr, output_np)
			
            fmax_val += fmax_
            aupr_val += aupr(y_csr.toarray().flatten(), output_np.flatten())

    model.eval()
    fmax_test = 0
    aupr_test = 0
    with torch.no_grad():
        for _, item in enumerate(test_data_loader):
            batch_x = process_batch_x(data['test_x'], item, data['shap_idx'], model).to(device)

            output = model(batch_x)

            output_np = torch.sigmoid(output).cpu().numpy()

            scores = np.zeros((len(test_data['test_pid_list']), len(test_data['mlb'].classes_)))
            scores[test_data['test_res_idx_']] = output_np
            format_output = output_format(test_data['test_pid_list'], test_data['mlb'].classes_, scores)
            (fmax_, t_), aupr_ = evaluate_metrics(test_data['pid_go'], format_output)
            fmax_test += fmax_
            aupr_test += aupr_

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train_b / total_train_iter),
          'loss_val: {:.4f}'.format(loss_val),
          'fmax_val: {:.4f}'.format(fmax_val),
          'aupr_val: {:.4f}'.format(aupr_val),
          'fmax_test: {:.4f}'.format(fmax_test),
          'aupr_test: {:.4f}'.format(aupr_test),
          )

    return loss_val, fmax_val, aupr_val, {'fmax_test': fmax_test, 'aupr_test': aupr_test}


def process_batch_x(features_list, batch_x_id, col_list, model):
    batch_x = []
    for features in features_list:
        x_csr_row_filter = features[batch_x_id.tolist()]
        x_csr_col_filter = x_csr_row_filter[:, col_list]
        batch_x.append(model.input(torch.from_numpy(x_csr_col_filter.indices).cuda().long(),
                                   torch.from_numpy(x_csr_col_filter.indptr).cuda().long(),
                                   torch.from_numpy(x_csr_col_filter.data).cuda().float()).unsqueeze(1))
    return torch.cat(batch_x, dim=1)


def process_batch_y(label_csr, batch_y_id):
    y_csr = label_csr[batch_y_id]
    batch_y = utils.sparse_mx_to_torch_sparse_tensor(y_csr).to_dense()
    return batch_y


def process_valid_batch_y(label_csr, batch_y_id):
    y_csr = label_csr[batch_y_id]
    batch_y = utils.sparse_mx_to_torch_sparse_tensor(y_csr).to_dense()
    return batch_y, y_csr


def evaluate_metrics(pid_go, pid_go_sc):
    pid_list = list(pid_go.keys())
    go_list = sorted(set(list(chain(*([pid_go[p_] for p_ in pid_list] +
                                      [pid_go_sc[p_] for p_ in pid_list if p_ in pid_go_sc])))) - ROOT_GO_TERMS)
    go_mat, score_mat = get_pid_go_mat(pid_go, pid_list, go_list), get_pid_go_sc_mat(pid_go_sc, pid_list, go_list)
    return utils.fmax(go_mat, score_mat), utils.pair_aupr(go_mat, score_mat)


def train():
    args = parse_args()

    device = args.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    back_dict, test_back_dict = load_protein_dataset_with_test_filter(path_root=args.root_path,
                                                                      dataset_name=args.dataset,
                                                                      hops=args.hops, t=args.t)
    train_size = back_dict['train_y'].shape[0]
    valid_size = back_dict['valid_y'].shape[0]
    test_size = back_dict['test_x'][0].shape[0]

    train_data_loader = Data.DataLoader(torch.arange(train_size), batch_size=args.batch_size, shuffle=True)
    val_data_loader = Data.DataLoader(torch.arange(valid_size), batch_size=10000)
    test_data_loader = Data.DataLoader(torch.arange(test_size), batch_size=10000)

    # model configuration
    model = TransformerModelOutputExcess(hops=args.hops,
                                         n_class=back_dict['train_y'].shape[1],
                                         input_dim=back_dict['train_x'][0].shape[1],
                                         n_layers=args.n_layers,
                                         num_heads=args.n_heads,
                                         hidden_dim=args.hidden_dim,
                                         dropout_rate=args.dropout,
                                         attention_dropout_rate=args.att_dropout).to(device)

    print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )
    fn_loss = nn.BCEWithLogitsLoss()

    t_total = time.time()
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    test_eval_all = []
    for epoch in range(args.epochs):
        loss_val, fmax_val, aupr_val, test_eval = train_valid_epoch(data=back_dict, test_data=test_back_dict,
                                                                    epoch=epoch, model=model,
                                                                    args=args, train_data_loader=train_data_loader,
                                                                    val_data_loader=val_data_loader,
                                                                    test_data_loader=test_data_loader,
                                                                    fn_loss=fn_loss, device=device, optimizer=optimizer,
                                                                    lr_scheduler=lr_scheduler)
        test_eval_all.append(test_eval)
        if early_stopping.check([fmax_val, aupr_val, loss_val], epoch):
            break

    print("Optimization Finished!")
    print("Train cost: {:.4f}s".format(time.time() - t_total))
    # Restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    print('Best metric fmax_val: ', str(early_stopping.best_vals[0]))
    print('Best metric aupr_val: ', str(early_stopping.best_vals[1]))
    print('Best metric loss_val: ', str(early_stopping.best_vals[2]))
    print('Best metric test: ', str(test_eval_all[int(early_stopping.best_epoch)]))
    model.load_state_dict(early_stopping.best_state)

	#todo
    output_name = F'/NAG_protein_data/best-DeepGraphGO-excess-filter-{args.dataset}-{get_time_stamp13()}.pt'
    print('Output Path: ', output_name)
    torch.save(model.state_dict(), output_name)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train()
