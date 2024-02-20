import sys
import numpy as np
import torch
import torch.cuda
import mindspore
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import functional as F
import mindspore.context as context
from mindspore.dataset import GeneratorDataset
import argparse
import os
from mindspore import load_checkpoint, save_checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn.optim import Adam
from mindspore import nn
from modules.GDGNNmodel import GDGNNModel
from dataPrepare.graph_data_concept_v4 import GraphDataset, MyData
from settings import *
from logger import Logger
import pandas as pd
import time
from torch_geometric.data import DataLoader, Data
from utils import *
# import ipdb
import random
# torch.set_default_tensor_type(torch.DoubleTensor)

# import torchsnooper
# import snoop
#
# torchsnooper.register_snoop()

import json
import math
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import cosine_similarity

clip_grad = 20.0
decay_epoch = 5
lr_decay = 0.8
max_decay = 5

ANNEAL_RATE = 0.00003


def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='News20')
    parser.add_argument('--model_type', type=str, default='GDGNNMODEL')
    parser.add_argument('--prior_type', type=str, default='Dir2')
    parser.add_argument('--enc_nh', type=int, default=128)
    parser.add_argument('--num_topic', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--init_mult', type=float, default=1.0)  # multiplier in initialization of decoder weight
    parser.add_argument('--device', default='cpu')  # do not use GPU acceleration
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--ni', type=int, default=300)  # 300
    parser.add_argument('--nw', type=int, default=300)

    parser.add_argument('--hop_num', type=int, default=1, help='the hop number')
    parser.add_argument('--edge_threshold', type=int, default=0, help='edge threshold')

    parser.add_argument('--fixing', action='store_true', default=False)
    parser.add_argument('--STOPWORD', action='store_true', default=True)
    parser.add_argument('--nwindow', type=int, default=5)
    parser.add_argument('--prior', type=float, default=0.5)
    parser.add_argument('--num_samp', type=int, default=1)
    parser.add_argument('--MIN_TEMP', type=float, default=0.3)
    parser.add_argument('--INITIAL_TEMP', type=float, default=1.0)
    parser.add_argument('--maskrate', type=float, default=0.5)

    parser.add_argument('--wdecay', type=float, default=1e-4)
    parser.add_argument('--word', action='store_true', default=True)
    parser.add_argument('--variance', type=float, default=0.995)  # default variance in prior normal in ProdLDA

    # gcn embedding
    # parser.add_argument('--use_gcn', type=bool, default=False, help='whether use gcn')
    parser.add_argument("--g_dim", type=int, default=100, help="embedding size of gcn embedding")
    parser.add_argument('--gcn_ratio', type=float, default=0.1, help="range=[0, 1], the weight value of recon loss")
    parser.add_argument('--gcn_epoch', type=int, default=2000, help='which epoch for rgcn to use')

    parser.add_argument('--use_td', action='store_true', default=False,
                        help='whether use topic diversity regularization')
    parser.add_argument('--use_recon', action='store_true', default=False, help='whether use gcn reconstruction')
    parser.add_argument('--td_ratio', type=float, default=0.1, help='the coefficient for topic diversity')

    parser.add_argument('--use_mr', action='store_true', default=False, help='whether use manifold regularization')
    parser.add_argument('--mr_ratio', type=float, default=0.1, help='the coefficient for maniflod regularization')
    parser.add_argument('--num_neigh', type=int, default=100,
                        help='the fraction of neighbors for maniflod regularization')

    parser.add_argument('--num_path', type=int, default=200, help='the number of shortest path')

    args = parser.parse_args()
    save_dir = ROOTPATH + "/models/%s/%s_%s/" % (args.dataset, args.dataset, args.model_type)
    opt_str = '_%s_m%.2f_lr%.4f' % (args.optimizer, args.momentum, args.learning_rate)

    seed_set = [1234, 2345, 3456, 4567, 5678, 6789, 7890]
    args.seed = seed_set[args.taskid]

    if args.model_type in ['GDGNNMODEL']:
        model_str = '_%s_ns%d_ench%d_ni%d_nw%d_hop%d_numpath%d_edgethres_%d_gcn_epoch_%d_temp%.2f-%.2f' % \
                    (args.model_type, args.num_samp, args.enc_nh, args.ni, args.nw, args.hop_num, args.num_path,
                     args.edge_threshold, args.gcn_epoch, args.INITIAL_TEMP, args.MIN_TEMP)
    else:
        raise ValueError("the specific model type is not supported")

    if args.model_type in ['GDGNNMODEL']:
        id_ = '%s_topic%d%s_prior_type%s_%.2f%s_%d_%d_stop%s_fix%s_word%s_td%s_recon%s_mr%s_%.2f_numneigh%.2f' % \
              (args.dataset, args.num_topic, model_str, args.prior_type, args.prior,
               opt_str, args.taskid, args.seed, str(args.STOPWORD), str(args.fixing), str(args.word), str(args.use_td),
               str(args.use_recon), str(args.use_mr), args.mr_ratio, args.num_neigh)
    else:
        id_ = '%s_topic%d%s%s_%d_%d_stop%s_fix%s_td%s_recon%s_mr%s_%.2f_numneigh%.2f' % \
              (args.dataset, args.num_topic, model_str,
               opt_str, args.taskid, args.seed, str(args.STOPWORD), str(args.fixing), str(args.use_td),
               str(args.use_recon),
               str(args.mr), args.mr_ratio, args.num_neigh)

    save_dir += id_
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    print("save dir", args.save_dir)
    args.save_path = os.path.join(save_dir, 'model.pt')
    # print("save path", args.save_path)

    args.log_path = os.path.join(save_dir, "log.txt")
    # print("log path", args.log_path)


    mindspore.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if 'cuda' in args.device:
        args.cuda = True
    else:
        args.cuda = False

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # if 'cuda' in args.device:
    #     args.cuda = True
    # else:
    #     args.cuda = False
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    return args


def test(model, test_loader, test_gcn_inputs, test_gcn_idxs, test_idx2sent, test_bow_batches, bs, data_len, mode='VAL',
         verbose=True):
    model.eval()  # switch to testing mode
    num_sent = 0
    val_output = {}
    for i, batch in enumerate(test_loader):
        gcn_input_batch = test_gcn_inputs[i]
        gcn_input_batch = [g.to(device) for g in gcn_input_batch]
        gcn_idx_batch = test_gcn_idxs[i]
        idx2sent_batch = test_idx2sent[i]
        # batch_size = batch.y.size(0)
        if not test_bow_batches is None:
            bow_batch = test_bow_batches[i]
            # print(i * bs)
            # print((i + 1) * bs)
            # print(data_len)
            batch_doc_ids = list(range(i * bs, np.min([(i + 1) * bs, data_len])))
        else:
            bow_batch = None
            batch_doc_ids = None

        batch = batch.to(device)
        batch_size = batch.y.shape(0)
        print_elements(gcn_input_batch)
        outputs = model.loss(batch, gcn_input_batch, gcn_idx_batch, idx2sent_batch, bow_batch, batch_doc_ids)
        # outputs = model.loss(batch, gcn_input_batch, gcn_idx_batch, idx2sent_batch, test_bow_batches)
        for key in outputs:
            if key not in val_output:
                val_output[key] = 0
            val_output[key] += outputs[key].item() * batch_size
        num_sent += batch_size

    if verbose:
        report_str = ' ,'.join(['{} {:.4f}'.format(key, val_output[key] / num_sent) for key in val_output])
        print('--{} {} '.format(mode, report_str))
    return val_output['loss'] / num_sent


def learn_feature(model, loader, test_gcn_inputs, test_gcn_idxs, test_idx2sent):
    model.eval()  # switch to testing mode
    thetas = []
    labels = []
    for i, batch in enumerate(loader):
        gcn_input_batch = test_gcn_inputs[i]
        gcn_input_batch = [g.to(device) for g in gcn_input_batch]
        gcn_idx_batch = test_gcn_idxs[i]
        idx2sent_batch = test_idx2sent[i]
        batch = batch.to(device)
        theta = model.get_doctopic(batch, gcn_input_batch, gcn_idx_batch, idx2sent_batch)
        thetas.append(theta)
        labels.append(batch.y)
    thetas = torch.cat(thetas, dim=0).detach()
    labels = torch.cat(labels, dim=0).detach()
    return thetas, labels


def eval_doctopic(model, test_loader, test_gcn_inputs, test_gcn_idxs, test_idx2sent):
    thetas, labels = learn_feature(model, test_loader, test_gcn_inputs, test_gcn_idxs, test_idx2sent)
    thetas = thetas.cpu().numpy()
    labels = labels.cpu().numpy()

    top_doctopic_results = eval_top_doctopic(thetas, labels)
    km_doctopic_results = eval_km_doctopic(thetas, labels)
    return top_doctopic_results, km_doctopic_results


def get_gcn_loader(batch_size, gcn_input, gcn_idx, idx2sent):
    # print(len(gcn_input))
    data_size = len(gcn_input)
    all_gcn_input_batches = []
    all_gcn_idx_batches = []
    all_idx2sent_batches = []

    for i in range(math.floor(data_size / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        all_gcn_input_batches.append(gcn_input[start:end])
        all_gcn_idx_batches.append(gcn_idx[start:end])
        all_idx2sent_batches.append(idx2sent[start:end])
    # the batch of which the length is less than batch_size
    rest = data_size % batch_size
    if rest > 0:
        all_gcn_input_batches.append(gcn_input[-rest:])
        all_gcn_idx_batches.append(gcn_idx[-rest:])
        all_idx2sent_batches.append(idx2sent[-rest:])

    return all_gcn_input_batches, all_gcn_idx_batches, all_idx2sent_batches


def get_bow_batches(bow_data, batch_size):
    all_bows_batches = []

    data_size = len(bow_data)
    for i in range(math.floor(data_size / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        all_bows_batches.append(bow_data[start:end])

    rest = data_size % batch_size
    if rest > 0:
        all_bows_batches.append(bow_data[-rest:])

    return all_bows_batches



# class MSData:
#     def __init__(self, data):
#         self.x = data.x.numpy()
#         self.edge_index = data.edge_index.numpy()
#         self.edge_attr = data.edge_attr
#         self.y = data.y.numpy()
#         self.pos = data.pos
#         self.normal = data.normal
#         self.face = data.face
#         self.edge_w = data.edge_w.numpy()
#         self.x_w = data.x_w.numpy()
#         self.edge_id = data.edge_id.numpy()
#         self.train = data.train.numpy()
#         self.graphy = data.graphy.numpy()

class MS_Batch_Data:
    def __init__(self, data):
        self.x = Tensor(data.x.numpy())
        edge_index=data.edge_index.numpy()
        if edge_index.size > 0:
            self.edge_index = Tensor(edge_index)
        else:
            self.edge_index = Tensor([-1])
        # self.edge_attr = data.edge_attr
        self.y = Tensor(data.y.numpy())
        # self.pos = data.pos
        # self.normal = data.normal
        # self.face = data.face
        self.edge_w = Tensor(data.edge_w.numpy())
        self.x_w = Tensor(data.x_w.numpy())
        self.edge_id = Tensor(data.edge_id.numpy())
        self.train = Tensor(data.train.numpy())
        self.graphy = Tensor(data.graphy.numpy())
        self.x_batch= Tensor(data.x_batch.numpy())
        self.y_batch = Tensor(data.x_batch.numpy())
        self.edge_id_batch = Tensor(data.edge_id_batch.numpy())
class MS_Input_Data:
    def __init__(self,data):
        self.edge_index=Tensor(data.edge_index.numpy())
        self.entity=Tensor(data.entity.numpy())
        self.edge_type=Tensor(data.edge_type.numpy())
        self.edge_norm=Tensor(data.edge_norm.numpy())
def print_elements(var):
    if isinstance(var, (list, tuple)):
        for elem in var:
            print(elem)
    elif isinstance(var, dict):
        for key, value in var.items():
            print(f'{key}: {value}')
    elif hasattr(var, '__dict__'):
        for attr, value in vars(var).items():
            # print(attr, type(value))
            print(f'{attr}: {value}')
    else:
        print(var)

def main(args):
    print("run here")
    global dataset, device
    print(args)

    # device = torch.device(args.device)

    device=context.set_context(device_target='CPU')
    path = todatapath(args.dataset, args.hop_num)
    args.path = path
    print(path)
    stop_str = '_stop' if args.STOPWORD else ''
    dataset = GraphDataset(path, hop_num=args.hop_num, STOPWORD=args.STOPWORD, edge_threshold=args.edge_threshold,
                           num_path=args.num_path)


    train_idxs = [i for i in range(len(dataset)) if dataset[i].train == 1]
    train_gcn_inputs=[]
    for idx in train_idxs:
        train_gcn_inputs.append(MS_Input_Data(dataset.all_gcn_inputs[idx]))
    # train_gcn_inputs = [MS_Input_Data(dataset.all_gcn_inputs[idx]) for idx in train_idxs]
    train_gcn_idxs = [dataset.all_gcn_idxs[idx] for idx in train_idxs]
    train_idx2sent = [dataset.all_idx2sent[idx] for idx in train_idxs]
    # print("train_gcn_inpus:")
    # print_elements(train_gcn_inpus[0])
    # exit(0)
    val_idxs = [i for i in range(len(dataset)) if dataset[i].train == -1]
    # val_idxs=val_idxs[:8]
    val_gcn_inputs = []
    for idx in val_idxs:
        val_gcn_inputs.append(MS_Input_Data(dataset.all_gcn_inputs[idx]))
    # val_gcn_inputs = [dataset.all_gcn_inputs[idx] for idx in val_idxs]
    val_gcn_idxs = [dataset.all_gcn_idxs[idx] for idx in val_idxs]
    val_idx2sent = [dataset.all_idx2sent[idx] for idx in val_idxs]

    test_idxs = [i for i in range(len(dataset)) if dataset[i].train == 0]
    # test_idxs=test_idxs[:8]
    test_gcn_inputs = []
    for idx in val_idxs:
        test_gcn_inputs.append(MS_Input_Data(dataset.all_gcn_inputs[idx]))
    # test_gcn_inputs = [dataset.all_gcn_inputs[idx] for idx in test_idxs]
    test_gcn_idxs = [dataset.all_gcn_idxs[idx] for idx in test_idxs]
    test_idx2sent = [dataset.all_idx2sent[idx] for idx in test_idxs]

    train_data = dataset[train_idxs]
    val_data = dataset[val_idxs]
    test_data = dataset[test_idxs]

    # MSData_lst = []
    # for i, data in enumerate(dataset):
    #     # print(i)
    #     MSData_lst.append(MSData(data))
    # # print("dataset type",type(dataset),dataset)
    # # train_idxs = train_idxs[:8]
    # print(type(dataset))
    # train_data = [MSData_lst[idx] for idx in train_idxs]
    # test_data = [MSData_lst[idx] for idx in test_idxs]
    # val_data = [MSData_lst[idx] for idx in val_idxs]
    # print(len(dataset.all_gcn_inputs))
    # print(dataset.all_gcn_inputs[0])
    # print(train_idxs[:5])
    # print(len(train_gcn_inputs))
    # print(train_gcn_inputs[0])

    vocab = dataset.vocab
    args.vocab = vocab
    args.vocab_size = len(vocab)

    print('data: %d samples  avg %.2f words' % (len(dataset), dataset.word_count / len(dataset)))
    print('finish reading datasets, vocab size is %d' % args.vocab_size)
    print('dropped sentences: %d' % dataset.dropped)
    print(len(train_data))

    # print_elements(train_data[0])
    # print("x shape:")
    # print(train_data[0].x.shape)
    # print(train_data[1].x.shape)
    # print(train_data[2].x.shape)
    # print(train_data[3].x.shape)

    # with open("ms_dataset.pkl", "wb") as f:
    #     pickle.dump(MSData_lst, f)
    # 定义一个生成器函数
    def generator_func(data):
        for item in data:
            yield (
            item.x, item.edge_index,  item.y, item.edge_w, item.x_w,
            item.edge_id, item.train, item.graphy)
            # yield (item.x,item.edge_index,item.edge_attr,item.y,item.pos,item.normal,item.face,item.edge_w,item.x_w,item.edge_id,item.train,item.graphy)
    log_niter = (len(train_data)) // 5
    args.iter_threahold = max(30 * len(train_data), 2000)
    # train_loader = GeneratorDataset(generator_func(train_data), shuffle=False,
    #                                 column_names=["x", "edge_index", "y",
    #                                               "edge_w", "x_w", "edge_id", "train", "graphy"])
    # test_loader = GeneratorDataset(generator_func(test_data), shuffle=False,
    #                                column_names=["x", "edge_index", "y",
    #                                              "edge_w", "x_w", "edge_id", "train", "graphy"])
    # val_loader = GeneratorDataset(generator_func(val_data), shuffle=False,
    #                               column_names=["x", "edge_index",  "y",
    #                                             "edge_w", "x_w", "edge_id", "train", "graphy"])
    # train_loader = train_loader.batch(1, drop_remainder=True)
    # test_loader = test_loader.batch(args.batch_size)
    # val_loader = val_loader.batch(args.batch_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, follow_batch=['x', 'edge_id', 'y'])
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, follow_batch=['x', 'edge_id', 'y'])
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, follow_batch=['x', 'edge_id', 'y'])
    # print("train_loader :")
    # for i, batch in enumerate(train_loader):
    #     print_elements(batch)
    #     exit(0)
    # print("train_loader: ")
    # print_elements(train_loader[0])
    # exit(0)

    train_gcn_inputs_batches, train_gcn_idxs_batches, train_idx2sent_batches = get_gcn_loader(args.batch_size,
                                                                                              train_gcn_inputs,
                                                                                              train_gcn_idxs,
                                                                                              train_idx2sent)
    # print(len(train_gcn_inputs_batches))
    val_gcn_inputs_batches, val_gcn_idxs_batches, val_idx2sent_batches = get_gcn_loader(args.batch_size, val_gcn_inputs,
                                                                                        val_gcn_idxs, val_idx2sent)
    test_gcn_inputs_batches, test_gcn_idxs_batches, test_idx2sent_batches = get_gcn_loader(args.batch_size,
                                                                                           test_gcn_inputs,
                                                                                           test_gcn_idxs, test_idx2sent)

    if args.use_mr:
        all_bows = []
        for d in dataset.ori_data:
            all_bows.append(onehot(np.array(d).astype('int'), args.vocab_size))

        train_bow_data = [all_bows[idx] for idx in train_idxs]
        val_bow_data = [all_bows[idx] for idx in val_idxs]
        test_bow_data = [all_bows[idx] for idx in test_idxs]

        # train_bow_batches = get_bow_batches(train_bow_data, args.batch_size)
        # val_bow_batches = get_bow_batches(val_bow_data, args.batch_size)
        # test_bow_batches = get_bow_batches(test_bow_data, args.batch_size)

        # all
        # ===========================   train   ================================
        train_bow_batch_array = np.stack(train_bow_data)  # train_docs, vocab_size
        s = dok_matrix(train_bow_batch_array, dtype=np.float64)
        sim_mat = cosine_similarity(s, s)  # train_docs, batch_size
        b_idx = np.argsort(sim_mat, axis=-1)[:, ::-1].copy()  # train_docs, batch_size
        train_doc_g = []
        for di in range(len(train_bow_data)):
            d = dict()
            for idx in b_idx[di, :args.num_neigh]:
                d[idx] = sim_mat[di, idx]
            train_doc_g.append(d)

        train_doc_g_batches = get_bow_batches(train_doc_g, args.batch_size)

        # ===========================   val   ================================
        val_bow_batch_array = np.stack(val_bow_data)  # val_docs, vocab_size
        s = dok_matrix(val_bow_batch_array, dtype=np.float64)
        sim_mat = cosine_similarity(s, s)  # val_docs, val_docs
        b_idx = np.argsort(sim_mat, axis=-1)[:, ::-1].copy()  # val_docs, val_docs
        val_doc_g = []
        for di in range(len(val_bow_data)):
            d = dict()
            for idx in b_idx[di, :args.num_neigh]:
                d[idx] = sim_mat[di, idx]
            val_doc_g.append(d)

        val_doc_g_batches = get_bow_batches(val_doc_g, args.batch_size)

        # ===========================   test   ================================
        test_bow_batch_array = np.stack(test_bow_data)  # test_docs, vocab_size
        s = dok_matrix(test_bow_batch_array, dtype=np.float64)
        sim_mat = cosine_similarity(s, s)  # test_docs, test_docs
        b_idx = np.argsort(sim_mat, axis=-1)[:, ::-1].copy()  # test_docs, test_docs
        test_doc_g = []
        for di in range(len(test_bow_data)):
            d = dict()
            for idx in b_idx[di, :args.num_neigh]:
                d[idx] = sim_mat[di, idx]
            test_doc_g.append(d)

        test_doc_g_batches = get_bow_batches(test_doc_g, args.batch_size)
    else:
        # all_bow_batches = None
        # train_bow_batches = None
        # val_bow_batches = None
        # test_bow_batches = None
        train_doc_g_batches = None
        val_doc_g_batches = None
        test_doc_g_batches = None

    whole_edge = dataset.whole_edge
    whole_edge_w = dataset.whole_edge_w
    # whole_edge = torch.tensor(whole_edge, dtype=torch.long, device=device)
    whole_edge = mindspore.Tensor(whole_edge, mindspore.int64)
    # print('edge number: %d' % whole_edge.size(1))

    save_edge(whole_edge.asnumpy(), whole_edge_w, vocab.id2word_, args.save_dir + '/whole_edge.csv')

    word_vec = np.load(path + '{}d_words{}.npy'.format(args.nw, dataset.stop_str))
    # word_vec = torch.from_numpy(word_vec).float()
    word_vec = mindspore.Tensor(word_vec).astype(mindspore.float32)
    if args.model_type == 'GDGNNMODEL':
        context.set_context(device_target="CPU")
        model = GDGNNModel(args, word_vec=word_vec, whole_edge=whole_edge)
    else:
        assert False, 'Unknown model type {}'.format(args.model_type)

    print(model)
    print('parameters', sum(param.numel() for param in model.trainable_params()))
    print('trainable parameters',sum(param.numel() for param in model.trainable_params() if param.requires_grad == True))
    # ======================================================  evaluation =============================================
    if args.eval:
        from mindspore import load_checkpoint, load_param_into_net
        param_dict = None
        args.temp = args.MIN_TEMP
        print('begin evaluation')
        if args.load_path != '':
            param_dict = load_checkpoint(args.load_path)
            # model.load_state_dict(torch.load(args.load_path, map_location=torch.device(device)))
            print("%s loaded" % args.load_path)
        else:
            param_dict = load_checkpoint(args.load_path)
            # model.load_state_dict(torch.load(args.save_path, map_location=torch.device(device)))
            print("%s loaded" % args.save_path)
        load_param_into_net(model, param_dict)
        model.eval()
        with torch.no_grad():
            if 'TMN' in args.dataset:
                refpath = todatapath('TMN', args.hop_num)
                data = pd.read_csv(refpath + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
            else:
                data = pd.read_csv(path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})

            common_texts = [text for text in data['content'].values]  # 使用的是internal reference corpus计算topic coherence
            beta = model.get_beta().detach().cpu().numpy()
            all_tc_results, all_td_results = eval_topic(beta, [vocab.id2word(i) for i in range(args.vocab_size)],
                                                        common_texts=common_texts)
            df_tc = pd.DataFrame(all_tc_results, index=['c_v', 'c_npmi', 'c_uci'], columns=[5, 10, 15, 20, 25])

            df_tc.to_csv(os.path.join(args.save_dir, 'tc.csv'))
            json.dump(all_td_results, open(os.path.join(args.save_dir, 'td.json'), 'w', encoding='utf-8'))

            if args.dataset in LABELED_DATASETS:
                top_doctopic_results, km_doctopic_results = eval_doctopic(model, test_loader, test_gcn_inputs_batches,
                                                                          test_gcn_idxs_batches, test_idx2sent_batches)
                df_dc = pd.DataFrame([top_doctopic_results, km_doctopic_results], index=['top', 'km'],
                                     columns=['purity', 'nmi'])
                df_dc.to_csv(os.path.join(args.save_dir, 'dc.csv'))

            if args.model_type in ['GDGNNMODEL']:
                beta_edge = model.get_beta_edge(False).detach().cpu().numpy()[:, 1:]
                print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=vocab.id2word_)
                for k in range(len(beta_edge)):
                    save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                              fname=args.save_dir + '/beta_edge_False_%d.csv' % k)
                beta_edge = model.get_beta_edge(True).detach().cpu().numpy()[:, 1:]
                print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=vocab.id2word_)
                for k in range(len(beta_edge)):
                    save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                              fname=args.save_dir + '/beta_edge_True_%d.csv' % k)
            if args.model_type in ['GDGNNMODEL']:
                W = model.get_W().detach().cpu().numpy()
                print('W', W)

        return
    # =================================================================================================================

    ALTER_TRAIN = True
    if args.optimizer == 'Adam':
        enc_optimizer = nn.Adam(model.enc_params, learning_rate=args.learning_rate, beta1=args.momentum,
                                weight_decay=args.wdecay)
        dec_optimizer = nn.Adam(model.dec_params, learning_rate=args.learning_rate, beta1=args.momentum,
                                weight_decay=args.wdecay)
    else:
        assert False, 'Unknown optimizer {}'.format(args.optimizer)

    best_loss = 1e4
    iter_ = decay_cnt = 0
    args.iter_ = iter_
    args.temp = args.INITIAL_TEMP
    opt_dict = {"not_improved": 0, "lr": args.learning_rate, "best_loss": 1e4}


    start = time.time()
    # args.iter_threahold = max(30 * len(train_loader), 2000)

    # ======================================================  training =============================================
    for epoch in range(args.num_epoch):
        num_sents = 0
        output_epoch = {}
        context.set_context(mode=context.PYNATIVE_MODE)
        # for i, batch in enumerate(train_loader.create_dict_iterator()):
        for i, batch_i in enumerate(train_loader):
            # print_elements(batch_i)
            batch=MS_Batch_Data(batch_i)
            # print_elements(batch)
            # exit(0)
            print("enter batch")
            gcn_inputs_batch = train_gcn_inputs_batches[i]
            # gcn_inputs_batch = [g.to(device) for g in gcn_inputs_batch]  # to gpu
            gcn_idxs_batch = train_gcn_idxs_batches[i]
            idx2sent_batch = train_idx2sent_batches[i]

            if args.use_mr:
                doc_g = train_doc_g_batches[i]
                if (i + 1) * args.batch_size <= len(train_bow_data):
                    batch_doc_ids = list(range(i * args.batch_size, (i + 1) * args.batch_size))
                else:
                    batch_doc_ids = list(range(i * args.batch_size, len(train_bow_data)))
            else:
                doc_g = None
                batch_doc_ids = None

            # batch = batch.to(device)
            print("before model loss:")
            batch_size = batch.y.shape[0]
            # print(batch_size)
            print_elements(gcn_inputs_batch[0])
            # exit(0)
            outputs = model.loss(batch, gcn_inputs_batch, gcn_idxs_batch, idx2sent_batch, doc_g, batch_doc_ids)
            loss = outputs['loss']
            num_sents += batch_size
            enc_optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)
            dec_optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

            # 清零梯度
            enc_optimizer.clear_grad()
            dec_optimizer.clear_grad()
            # 反向传播
            loss.backward()
            # 梯度裁剪
            clip_value=1.0
            clip_grad = P.ClipByValue(max=clip_value)
            model_params = model.trainable_params()
            grads = [param.grad for param in model_params]
            clipped_grads = [clip_grad(grad) for grad in grads]
            for param, grad in zip(model_params, clipped_grads):
                param.set_grad(grad)



            if ALTER_TRAIN:
                if epoch % 2 == 0:
                    dec_optimizer.step()
                else:
                    enc_optimizer.step()
            else:
                enc_optimizer.step()
                dec_optimizer.step()

            # report
            for key in outputs:
                if key not in output_epoch:
                    output_epoch[key] = 0
                output_epoch[key] += outputs[key].item() * batch_size

            if iter_ % log_niter == 0:
                report_str = ' ,'.join(['{} {:.4f}'.format(key, output_epoch[key] / num_sents) for key in output_epoch])
                print(
                    'Epoch {}, iter {}, {}, time elapsed {:.2f}s'.format(epoch, iter_, report_str, time.time() - start))
            iter_ += 1
            args.iter_ = iter_
            ntither = args.iter_ - args.iter_threahold

            if ntither >= 0 and ntither % 1000 == 0 and args.temp > args.MIN_TEMP:
                args.temp = max(args.temp * math.exp(- ANNEAL_RATE * ntither), args.MIN_TEMP)
                best_loss = 1e4
                opt_dict["best_loss"] = best_loss
                opt_dict["not_improved"] = 0
                model.load_state_dict(torch.load(args.save_path))

        if ALTER_TRAIN and epoch % 2 == 0:
            continue

        model.set_train(mode=False)  # 等价于 PyTorch 中的 model.eval()

        val_loss = test(model, val_loader, val_gcn_inputs_batches, val_gcn_idxs_batches, val_idx2sent_batches,
                        val_doc_g_batches, args.batch_size, len(val_bow_data), 'VAL')

        model.set_train(mode=True)  # 将模型设置回训练模式
        print(best_loss, opt_dict["best_loss"], args.temp, ALTER_TRAIN)

        i # 检查是否有改进
        if val_loss < best_loss:
            print('update best loss')
            best_loss = val_loss
            save_checkpoint(model, args.save_path)
        if val_loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epoch >= 15 and args.temp == args.MIN_TEMP:
                opt_dict["best_loss"] = best_loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                from mindspore import  load_checkpoint
                load_checkpoint(args.save_path, net=model)
                print('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                if args.optimizer == 'Adam':
                    enc_optimizer = Adam(model.enc_params, learning_rate=opt_dict["lr"], weight_decay=args.wdecay)
                    dec_optimizer = Adam(model.dec_params, learning_rate=opt_dict["lr"], weight_decay=args.wdecay)
                else:
                    assert False, 'Unknown optimizer {}'.format(args.optimizer)
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = val_loss
        if decay_cnt == max_decay:
            break
        model.set_train(mode=False)
        test(model, test_loader, test_gcn_inputs_batches, test_gcn_idxs_batches, test_idx2sent_batches,
             test_doc_g_batches, args.batch_size, len(test_bow_data), 'TEST')

        if epoch % 5 == 0:
            beta = model.get_beta().asnumpy()
            if epoch > 0 and (epoch) % 50 == 0:
                data = pd.read_csv(path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
                common_texts = [text for text in data['content'].values]
                eval_topic(beta, [vocab.id2word(i) for i in range(args.vocab_size)], common_texts=common_texts)
                if args.model_type in ['GDGNNMODEL']:
                    beta_edge = model.get_beta_edge(False).asnumpy()[:, 1:]
                    print_top_pairwords(beta_edge, edge_index=whole_edge.asnumpy(), vocab=vocab.id2word_)
                    for k in range(len(beta_edge)):
                        save_edge(whole_edge.asnumpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                                  fname=args.save_dir + '/beta_edge_False_%d.csv' % k)
                    beta_edge = model.get_beta_edge(True).asnumpy()[:, 1:]
                    print_top_pairwords(beta_edge, edge_index=whole_edge.asnumpy(), vocab=vocab.id2word_)
                    for k in range(len(beta_edge)):
                        save_edge(whole_edge.asnumpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                                  fname=args.save_dir + '/beta_edge_True_%d.csv' % k)
                if args.model_type in ['GDGNNMODEL5']:
                    W = model.get_W().asnumpy()  # M
                    print('W', W)
            else:
                eval_topic(beta, [vocab.id2word(i) for i in range(args.vocab_size)])

            if args.dataset in LABELED_DATASETS:
                eval_doctopic(model, test_loader, test_gcn_inputs_batches, test_gcn_idxs_batches,
                              test_idx2sent_batches)

        model.set_train(mode=True)

    # 加载模型状态
    from mindspore import  load_checkpoint
    load_checkpoint(args.save_path, net=model)

    # 将模型设置为评估模式
    model.set_train(mode=False)
    # 训练和验证循环
    for epoch in range(args.epochs):
        # 训练步骤...

        # 验证步骤
        model.set_train(mode=False)
        test(model, test_loader, test_gcn_inputs_batches, test_gcn_idxs_batches, test_idx2sent_batches,
             test_doc_g_batches, args.batch_size, len(test_bow_data), 'TEST')

        if 'TMN' in args.dataset:
            refpath = todatapath('TMN', args.hop_num)
            data = pd.read_csv(refpath + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
        else:
            data = pd.read_csv(path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
        common_texts = [text for text in data['content'].values]
        beta = model.get_beta().asnumpy()

        all_tc_results, all_td_results = eval_topic(beta, [vocab.id2word(i) for i in range(args.vocab_size)],
                                                    common_texts=common_texts)
        df_tc = pd.DataFrame(all_tc_results, index=['c_v', 'c_npmi', 'c_uci'], columns=[5, 10, 15, 20, 25])

        df_tc.to_csv(os.path.join(args.save_dir, 'tc.csv'))
        json.dump(all_td_results, open(os.path.join(args.save_dir, 'td.json'), 'w', encoding='utf-8'))

        if args.dataset in LABELED_DATASETS:
            top_doctopic_results, km_doctopic_results = eval_doctopic(model, test_loader, test_gcn_inputs_batches,
                                                                      test_gcn_idxs_batches, test_idx2sent_batches)
            df_dc = pd.DataFrame([top_doctopic_results, km_doctopic_results], index=['top', 'km'],
                                 columns=['purity', 'nmi'])
            df_dc.to_csv(os.path.join(args.save_dir, 'dc.csv'))

        if args.model_type in ['GDGNNMODEL']:
            beta_edge = model.get_beta_edge(False).asnumpy()[:, 1:]
            print_top_pairwords(beta_edge, edge_index=whole_edge.asnumpy(), vocab=vocab.id2word_)
            for k in range(len(beta_edge)):
                save_edge(whole_edge.asnumpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                          fname=args.save_dir + '/beta_edge_False_%d.csv' % k)
            beta_edge = model.get_beta_edge(True).asnumpy()[:, 1:]
            print_top_pairwords(beta_edge, edge_index=whole_edge.asnumpy(), vocab=vocab.id2word_)
            for k in range(len(beta_edge)):
                save_edge(whole_edge.asnumpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                          fname=args.save_dir + '/beta_edge_True_%d.csv' % k)
        if args.model_type in ['GDGNNMODEL']:
            W = model.get_W().asnumpy()
            print('W', W)

    model.set_train(mode=True)


if __name__ == '__main__':
    args = init_config()
    print(args.eval)
    if not args.eval:
        print("in if ")
        sys.stdout = Logger(args.log_path)
    print("out if ")
    main(args)
