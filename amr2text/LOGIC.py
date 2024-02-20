import torch

root = 0
graph_dict = {}

def tokenize(line):
    if line[0] != '(':
        line = ['('] + line + [')']
    for word in line:
        if word in ['.', '_', ',']:
            continue
        yield word


def parse_inner(toks):
    children = []
    while True:
        word = next(toks)
        if word == '(':
            if len(children) > 0:
                parent = children.pop()
            else:
                parent = ''
            children.append((parent, parse_inner(toks)))
        elif word == ')':
            return children
        else:
            children.append(word)


def parse_root(toks):
    word = next(toks)
    if word != '(':
        print('error')
    return ('root', parse_inner(toks))


def postprocessing(tree):
    i = 0
    new_tree = []
    # 先合并exist，把exist和它后面的一个节点合并成一个新节点，其他节点暂时不处理
    while i < len(tree):
        node = tree[i]
        if node in ['exists', '-exists', '--exists', '-all', 'all']:
            new_node = (node, [tree[i+1]])
            new_tree.append(new_node)
            i += 2
        elif node == '-' and i + 1 < len(tree):
            new_node = (node, [tree[i+1]])
            new_tree.append(new_node)
            i += 2            
        else:
            new_tree.append(node)
            i += 1

    i = 0
    tree = new_tree.copy()
    new_tree = []
    # 合并逻辑关系符号，如“&”,"|"等等，将逻辑关系符号前后两个节点放到逻辑关系符号下面当子树。
    while i < len(tree):
        node = tree[i]
        if node in ['=', '&', '|', '->', '-']:
            left_node = new_tree.pop()
            right_node = tree[i+1]
            new_node = (node, ([left_node, right_node]))
            new_tree.append(new_node)
            i += 2
        else:
            new_tree.append(node)
            i += 1
    
    i = 0
    tree = new_tree.copy()
    new_tree = []
    # 递归子树
    while i < len(tree):    
        node = tree[i]
        if isinstance(node, tuple):
            new_tree.append((node[0], postprocessing(node[1])))
        else:
            new_tree.append(node)
        i += 1

    return new_tree


def transfer(tree):
    global root
    root = 1
    g = [[], []]  # traverse, parent
    first_label = tree[0]
    g[0].append(first_label)
    g[1].append(0)  # 根节点的parent为0
    _sub(g, tree[1], root)
    return g[0], g[1]


def _sub(g, tree, inc):
    global root

    for i in tree:
        if isinstance(i, tuple):
            root = root + 1
            g[0].append(i[0])
            g[1].append(inc)
            _sub(g, i[1], root)
        else:
            root = root + 1
            g[0].append(i)
            g[1].append(inc)


def need_reduce(node):
    return (node.startswith('e') or node.startswith('x')) and node[1:].isdigit()


# def need_edge(node, inc_name):
#     if inc_name == '&':
#         if node in ['exists', '-exists', '--exists', '-all', 'all', '|', '->', '-']:
#             return True
#         else:
#             return False
#     elif node == '=':
#         return False
#     else:
#         return True

def need_edge(node, inc_name):
    return True


# def need_node(node_name, inc_name, i):
#     if (node_name == inc_name and node_name in ['=', '&', '|', '->', '-']):  # 非两个连续相同的关系节点
#         return False
#     else:
#         return True


def need_node(node_name, inc_name, i):
    return True

def make_graph_return(g):
    traverse = []
    edges = []
    for node in g[0]:
        traverse.append(node[1])
    for edge in g[1]:
        edges.append((edge[0]))
    return traverse, edges

def transfer_as_graph(tree):
    global root
    global graph_dict
    root = 0
    graph_dict = {}
    g = [[], []]
    # g = Digraph("G", filename=name, format='png', strict=True)
    first_label = tree[0]
    g[0].append((0, first_label))
    # g.node("0", first_label)
    graph_dict[first_label] = root
    _sub_graph(g, tree[1], 0, first_label)
    # final_g = delete_node(g)
    # return make_graph_return(final_g)
    return make_graph_return(g)


def count_edge(g):
    edge_counter = {}
    for node in g[0]:
        edge_counter[node[0]] = 0
    for edge in g[1]:
        edge_counter[edge[0][0]] += 1
        edge_counter[edge[0][1]] += 1
    return edge_counter


def delete_node(g):
    edge_counter = count_edge(g)
    final_node = []
    for node in g[0]:
        if node[1] == '&' and edge_counter[node[0]] == 1:
            for index, edge in enumerate(g[1]):
                if node[0] in edge[0]:
                    g[1].pop(index)
                    break
        # else:
        final_node.append(node)

    return [final_node, g[1]]


def _sub_graph(g, tree, inc, inc_name):
    global root
    global graph_dict
    for i in tree:
        if isinstance(i, tuple):
            node_name = i[0]
        else:
            node_name = i
        if need_reduce(node_name):  # 是否是实体如x01或者事件如e01
            if node_name in graph_dict.keys():  # 判断这个x或e是否已经出现过，出现过则用原来的节点编号即可
                exist_root = graph_dict[node_name]
                if need_edge(node_name, inc_name):
                    # g.edge(exist_root, inc, '')
                    # g[1].append(((exist_root, inc), ''))
                    g[1].append(((inc, exist_root), ''))
            else:
                root = root + 1
                if node_name not in graph_dict.keys():
                    graph_dict[node_name] = root
                # g.node(root, node_name)
                g[0].append((root, node_name))
                if need_edge(node_name, inc_name):
                    # g.edge(inc, root, '')
                    g[1].append(((inc, root), ''))

        else:
            if need_node(node_name, inc_name, i):
                root = root + 1
                # g.node(root, node_name)
                g[0].append((root, node_name))
                if need_edge(node_name, inc_name):
                    # g.edge(inc, root, '')
                    g[1].append(((inc, root), ''))
        if isinstance(i, tuple):
            _sub_graph(g, i[1], root, i[0])


class LOGICdata:
    def __init__(self, words, traverse, parents_or_edges, with_reentrancies):
        self.idx = 0
        self.annotation = " ".join(words)
        self.traverse = traverse

        self.parents_or_edges = parents_or_edges
        # [第一个根节点的父节点为 0，剩下的节点的父节点对应列表index+1，不存在则为-1]
        self.matrix = torch.IntTensor(3, len(self.traverse), len(self.traverse)).zero_()
        # 第二个矩阵记录的是第i行的点的子孙
        # 第三个矩阵记录的是第i行的点的父亲
        # tensor([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
        # tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)

        self.matrix[0, :, :] = torch.eye(len(self.traverse))
        if with_reentrancies:
            # 此时是edges
            for index, p in enumerate(self.parents_or_edges):
                self.matrix[1, p[0], p[1]] = 1
                # self.matrix[1, p[1], p[0]] = 1

                self.matrix[2, p[1], p[0]] = 1
                # self.matrix[2, p[0], p[1]] = 1
            self.parents_or_edges = list(range(len(self.traverse)))
        else:
            # 此时是parents
            for index, p in enumerate(self.parents_or_edges):
                if p == 0:
                    continue
                self.matrix[1, p-1, index] = 1
                self.matrix[2, index, p-1] = 1
        
        # print(self.traverse)
        # print(self.parents_or_edges)
        # print(self.matrix[1])
        # print(self.matrix[2])
        # fsdf

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.annotation)
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.traverse)
    
    def __getitem__(self, key):
        return self.traverse[key]

    def __next__(self):
        self.idx += 1
        try:
            word = self.traverse[self.idx - 1]
            return word
        except IndexError:
            self.idx = 0
            raise StopIteration
    next = __next__

def extract_logic_features(line, with_reentrancies):
    if not line:
        return [], []
    words = tuple(line)
    tree = parse_root(tokenize(line))
    new_tree = postprocessing(tree[1])
    new_tree = ('root', new_tree)
    if not with_reentrancies:
        traverse, parents_or_edges = transfer(new_tree)
    else:
        traverse, parents_or_edges = transfer_as_graph(new_tree)
    # print(traverse, parents_or_edges)
    return LOGICdata(words, traverse, parents_or_edges, with_reentrancies)


# line = 'exists x01 . ( _ woman ( x01 ) & exists e02 . ( _ quarrel ( e02 ) & ( subj ( e02 ) = x01 ) & exists x03 . ( _ husband ( x03 ) & _ with ( e02 , x03 ) ) & exists x04 . ( _ shirt ( x04 ) & _ blue ( x04 ) & _ in ( e02 , x04 ) ) ) )'



# extract_logic_features(line.split(), True)