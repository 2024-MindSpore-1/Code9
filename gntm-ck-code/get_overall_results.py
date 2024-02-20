import argparse
from settings import *
import pandas as pd
import os
import json
import numpy as np

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='News20')
    parser.add_argument('--model_type', type=str, default='GDGNNMODEL')
    parser.add_argument('--prior_type', type=str, default='Dir2')
    parser.add_argument('--enc_nh', type=int, default=128)
    # parser.add_argument('--num_topic', type=int, default=20)
    # parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--num_epoch', type=int, default=10)
    # parser.add_argument('--init_mult', type=float, default=1.0)  # multiplier in initialization of decoder weight
    # parser.add_argument('--device', default='cpu')  # do not use GPU acceleration
    # parser.add_argument('--eval', action='store_true', default=False)
    # parser.add_argument('--taskid', type=int, default=0, help='slurm task id')    
    # parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--ni', type=int, default=300) # 300
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
    # parser.add_argument('--maskrate', type=float, default=0.5)

    # parser.add_argument('--wdecay', type=float, default= 1e-4)
    parser.add_argument('--word', action='store_true', default= True)
    # parser.add_argument('--variance', type=float, default=0.995)  # default variance in prior normal in ProdLDA
    parser.add_argument('--gcn_epoch', type=int, default=2000, help='which epoch for rgcn to use')
    parser.add_argument('--use_td', action='store_true', default= False, help='whether use topic diversity regularization')
    parser.add_argument('--use_recon', action='store_true', default= False, help='whether use gcn reconstruction')

    parser.add_argument('--td_ratio', type=float, default=0.1, help='the coefficient for topic diversity')

    parser.add_argument('--use_mr', action='store_true', default= False, help='whether use manifold regularization')
    parser.add_argument('--mr_ratio', type=float, default=0.1, help='the coefficient for maniflod regularization')
    parser.add_argument('--num_neigh', type=int, default=50, help='the number of neighbors for maniflod regularization')

    parser.add_argument('--num_path', type=int, default=200, help='the number of shortest path')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args= init_config()
    opt_str = '_%s_m%.2f_lr%.4f' % (args.optimizer, args.momentum, args.learning_rate)
    save_dir = "./models/%s/%s_%s/" % (args.dataset, args.dataset, args.model_type)
    seed_set = [1234, 2345, 3456, 4567, 5678, 6789, 7890]

    if args.model_type in ['GDGNNMODEL']:
        model_str = '_%s_ns%d_ench%d_ni%d_nw%d_hop%d_numpath%d_edgethres_%d_gcn_epoch_%d_temp%.2f-%.2f' % \
                    (args.model_type, args.num_samp, args.enc_nh, args.ni, args.nw, args.hop_num, args.num_path, 
                     args.edge_threshold, args.gcn_epoch, args.INITIAL_TEMP, args.MIN_TEMP)
    else:
        raise ValueError("the specific model type is not supported")

    all_topics = [20, 30, 50]

    all_runs = [0, 1, 2, 3, 4]
    all_topns = [5, 10, 15, 20, 25]

    all_tc_results = []

    for i, topic in enumerate(all_topics):
        tc_result = []
        dc_result = []
        for run in all_runs:
            seed = seed_set[run]
            if args.model_type in [ 'GDGNNMODEL']:
                id_ = '%s_topic%d%s_prior_type%s_%.2f%s_%d_%d_stop%s_fix%s_word%s_td%s_recon%s_mr%s_%.2f_numneigh%.2f' % \
                (args.dataset, topic, model_str, args.prior_type, args.prior,
                opt_str, run, seed, str(args.STOPWORD), str(args.fixing), str(args.word), str(args.use_td), 
                str(args.use_recon), str(args.use_mr), args.mr_ratio, args.num_neigh)
            else:
                id_ = '%s_topic%d%s%s_%d_%d_stop%s_fix%s_td%s_recon%s_mr%s_%.2f_numneigh%.2f' % \
                (args.dataset, args.num_topic, model_str,
               opt_str, run, seed, str(args.STOPWORD), str(args.fixing), str(args.use_td), 
               str(args.use_recon), str(args.use_mr), args.mr_ratio, args.num_neigh)

            save_dir_temp = save_dir + id_

            print(save_dir_temp)

            if os.path.exists(save_dir_temp):
                print('good')
            else:
                print('bad')

            tc_results = pd.read_csv(os.path.join(save_dir_temp, 'tc.csv'))    # csv
            for tc in tc_results.values[:, 1:]:
                tc_result.extend(tc)

            td_results = json.load(open(os.path.join(save_dir_temp, 'td.json'), 'r', encoding='utf-8'))    # list
            tc_result.extend(td_results)

            all_tc_results.append(tc_result)

            # dc_results = pd.read_csv(os.path.join(save_dir_temp, 'dc.csv'))
            # for dc in dc_results.values[:, 1:].transpose():
            #     dc_result.extend(dc)

            # all_dc_results.append(dc_result)
            
            tc_result = []
            # dc_result = []

        all_tc_results.append(np.mean(all_tc_results[i*5 + i*2:(i+1)*5 + i*2], axis=0))
        all_tc_results.append(np.std(all_tc_results[i*5 + i*2:(i+1)*5 + i*2], axis=0))

        # all_dc_results.append(np.mean(all_dc_results[i*5 + i*2:(i+1)*5 + i*2], axis=0))
        # all_dc_results.append(np.std(all_dc_results[i*5 + i*2:(i+1)*5 + i*2], axis=0))

    # index
    tc_row_indexs = []
    for K in all_topics:
        tc_row_indexs.extend(list(zip([K] * len(all_runs + ['mean', 'std']), all_runs + ['mean', 'std'])))

    # column
    tc_column_indexs = []
    for coh_type in ['c_v', 'c_npmi', 'c_uci', 'td']:
        tc_column_indexs.extend(list(zip([coh_type] * len(all_topns), all_topns)))

    tc_mindex = pd.MultiIndex.from_tuples(tc_row_indexs, names=['K', 'round'])
    tc_nindex = pd.MultiIndex.from_tuples(tc_column_indexs, names=['type', 'N'])

    df = pd.DataFrame(all_tc_results, index=tc_mindex, columns=tc_nindex)
    df.to_excel('final/topic_coherence_gntm_concept_%s_td%s_recon%s_mr%s_%f_numneigh%f_edgethres%d_gcnepoch%d_hop%d_numpath%d.xlsx' % (args.dataset, args.use_td, args.use_recon, args.use_mr, args.mr_ratio, args.num_neigh, args.edge_threshold, args.gcn_epoch, args.hop_num, args.num_path))

    # index
    # dc_row_indexs = []
    # for K in all_topics:
    #     dc_row_indexs.extend(list(zip([K] * len(all_runs + ['mean', 'std']), all_runs + ['mean', 'std'])))

    # # column
    # dc_column_indexs = []
    # for coh_type in ['purity', 'nmi']:
    #     dc_column_indexs.extend(list(zip([coh_type] * len(['top', 'km']), ['top', 'km'])))

    # dc_mindex = pd.MultiIndex.from_tuples(dc_row_indexs, names=['K', 'round'])
    # dc_nindex = pd.MultiIndex.from_tuples(dc_column_indexs, names=['metric', 'type'])

    # df = pd.DataFrame(all_dc_results, index=dc_mindex, columns=dc_nindex)
    # df.to_excel('doc_clustering_gntm_concept_%s_td%s_recon%s_mr%s_%f_numneigh%f_edgethres%d_gcnepoch%d_hop%d_numpath%d.xlsx' % (args.dataset, args.use_td, args.use_recon, args.use_mr, args.mr_ratio, args.num_neigh, args.edge_threshold, args.gcn_epoch, args.hop_num, args.num_path))
