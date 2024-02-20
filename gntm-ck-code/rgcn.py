import math
import numpy as np
from typing import List, Optional, Set
import mindspore
import mindspore.numpy as mnp
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform, initializer,Normal
from mindspore.ops import operations as P
from mindspore import Parameter,Tensor,ops


from mindspore.common.initializer import Uniform
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        minval = Tensor(-bound, mindspore.float32)
        maxval = Tensor(bound, mindspore.float32)
        tensor.set_data(ops.uniform(tensor.shape, minval, maxval, seed=5))
class RGCN(nn.Cell):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = Parameter(initializer(XavierUniform(), [num_relations, 100]))

        self.conv1 = RGCNConv(100, 100, num_relations * 2, num_bases)
        self.conv2 = RGCNConv(100, 100, num_relations * 2, num_bases)
        self.dropout_ratio = dropout
        self.relu=P.ReLU()
        self.dropout = nn.Dropout(keep_prob=1 - dropout)


    def construct(self, entity, edge_index, edge_type, edge_norm):
        # print("RGCN construct")
        # print(entity.shape)
        # exit(0)
        x = self.entity_embedding(entity)

        x = self.conv1(x, edge_index, edge_type, edge_norm)
        x = self.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = P.ReduceSum()(s * r * o, 1)
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        return score, P.SigmoidCrossEntropyWithLogits()(score, target)

    def reg_loss(self, embedding):
        return P.ReduceMean()(embedding**2) + P.ReduceMean()(self.relation_embedding**2)

class RGCNConv(nn.Cell):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True):
        super(RGCNConv, self).__init__()
        self.matmul = ops.MatMul()
        self.add = ops.TensorAdd()
        self.mul = ops.Mul()
        self.unsqueeze = ops.ExpandDims()
        self.squeeze = ops.Squeeze(2)
        self.index_select = ops.GatherV2()
        self.bias_add = ops.BiasAdd()
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.scatter_add=ops.ScatterAdd()
        self.softmax=nn.Softmax()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_l = Parameter(Tensor(out_channels, mindspore.float32), name="att_l")
        self.att_r = Parameter(Tensor(out_channels, mindspore.float32), name="att_r")
        self.lin_l = nn.Dense(in_channels, out_channels)

        self.num_relations = num_relations
        self.num_bases = num_bases
        self.basis = Parameter(initializer(XavierUniform(), [num_bases, in_channels, out_channels]))
        self.att = Parameter(initializer(XavierUniform(), [num_relations, num_bases]))
        # self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        # self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = Parameter(initializer(XavierUniform(), [in_channels, out_channels]))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(initializer(Normal(0,0.02), [out_channels]))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def construct(self, x, edge_index, edge_type, edge_norm=None):
        return self.propagate(edge_index, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def propagate(self, edge_index,x, edge_type, edge_norm):
        # Step 1: Linearly transform node feature matrix.
        x = self.lin_l(x)
        # Step 2: Compute attention coefficients.
        alpha_l = self.reduce_sum(x[edge_index[0]] * self.att_l,0)
        alpha_r = self.reduce_sum(x[edge_index[1]] * self.att_r,0)
        alpha = self.leaky_relu(alpha_l + alpha_r)
        # Step 3: Normalize attention coefficients.
        alpha = self.softmax(alpha)

        # Step 4: Weighted feature matrix.
        x_j = x[edge_index[1]]
        out = self.message(x_j, edge_index[1], edge_type, edge_norm)
        # Step 5: Sum up weighted incoming messages.
        # 创建一个形状为(100,)的索引张量
        out_indices_tensor = Tensor(edge_index[1], mindspore.int32)

        # 创建一个形状为(100, 100)的零张量
        out_tensor = Tensor(np.zeros((100, 100)), mindspore.float32)

        # 使用ScatterAdd操作更新out_tensor
        out_tensor = self.scatter_add(out_tensor, out_indices_tensor, out)


        # Step 6: Update node embeddings.
        out = self.update(out_tensor, x)

        return out

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):

        w = self.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j is None:
            print("x_j none")
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = self.index_select(w, 0, index)
        else:
            print("x_j not none ")
            print(w)
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            print(w)
            print("before gather")
            indices=0
            print(edge_type)
            axis=0
            axis_tensor = Tensor(axis, mindspore.int32)
            indices_tensor = Tensor(indices, mindspore.int32)
            w = self.index_select(w, indices_tensor, axis_tensor)
            print(w)
            print("after gather")
            out=self.matmul(x_j, w)
            # out = self.squeeze(self.matmul(self.unsqueeze(x_j, 1), w))


        return out if edge_norm is None else self.mul(out, self.unsqueeze(edge_norm, -1))



    def update(self, aggr_out, x):
        lenth=len(x[0].shape)
        print(x.shape)
        if self.root is not None:
            if x is None:
                print("x none")
                aggr_out = aggr_out + self.root
            else:
                print("x_not_none")
                x_root=self.matmul(x, self.root)[:100,:]
                aggr_out = aggr_out + x_root
                exit(0)
                # aggr_out = aggr_out + self.matmul(x, self.root)

        if self.bias is not None:
            aggr_out = self.bias_add(aggr_out,self.bias)
        return aggr_out


    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
