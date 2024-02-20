def tokenize(line):
    if not line.startswith('('):
        line = '( ' + line + ' )'
    for word in line.split():
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


# line = 'exists x01 . ( _ man ( x01 ) & exists x02 . ( _ luddite ( x02 ) & ( x01 = x02 ) ) & - _ would ( exists x03 . ( _ pc ( x03 ) & exists e04 . ( _ own ( e04 ) & ( subj ( e04 ) = x01 ) & ( acc ( e04 ) = x03 ) ) ) ) )'
# tree = parse_root(tokenize(line))
# # print(tree[1])
# new_tree = postprocessing(tree[1])
# new_tree = ('root', new_tree)
# print(new_tree)
# dsf

file_path_list = ['data/src-train_post.txt', 'data/src-val_post.txt', 'data/src-test_post.txt']

for file_path in file_path_list:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    pair = []
    for line in lines:
        try:
            tree = parse_root(tokenize(line))
            new_tree = postprocessing(tree[1])
            new_tree = ('root', new_tree)
            pair.append('[source]{}[tree]{}'.format(line, new_tree))
        except:
            print(line)
            fsdf

    with open(file_path.replace('.txt', '_tree.txt'), 'w', encoding='utf-8') as f:
        for line in pair:
            f.write(line + '\n')
