hop_num=1          # specify the hop number H, choices in {1, 2}
edge_threshold=0
gcn_epoch=2000
num_path=100       # specify the maximum number of pairs P, choices in {50, 100}
mr_ratio=0.01      # specify the manifold coefficient \lambda, choices in {0.01, 0.1}
num_neigh=100      # specify the number of nearest neighbors R, choices in {50, 100}

python get_overall_results.py \
    --dataset Reuters \
    --model GDGNNMODEL \
    --num_epoch 400 \
    --ni 300  \
    --hop_num ${hop_num} \
    --edge_threshold ${edge_threshold} \
    --word \
    --gcn_epoch ${gcn_epoch} \
    --use_mr \
    --mr_ratio ${mr_ratio} \
    --num_neigh ${num_neigh} \
    --num_path ${num_path} \
    --nwindow 5
