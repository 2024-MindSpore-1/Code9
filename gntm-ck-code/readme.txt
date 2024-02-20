=======================================    How to run   =============================================

[1] Setup the environment: 
    conda env create -f environment.yml
    (optional) pip install requirements.txt

[2] Set the ROOTPATH in file 'settings.py' to be the absolute path of this code

[3] All the data files are saved in the folders 'data/reuters/' and 'data/reuters_2hop/'.

    The file 'preprocess.py' in the folder 'dataPrepare/' is used to remove the stop words 
    and build the vocabulary in the raw reuters dataset.

    The triples extracted from ConceptNet for each doc are saved in those files with the names 
    prefixed 'all_doc_triples_'. The word pairs that has some commonsense relationship in each
    doc are saved in those files with the names prefixed 'all_doc_pairs_'. 

    To see how to represent the documents into graphs in the code, see the file 'graph_data_concept.py' 
    in the folder 'dataPrepare/'.

[4] Pretrain the rgcn using the script 'run_reuters_rgcn_pretrain_xhop.sh' (x=1 or x=2)
    ```
    bash run_reuters_rgcn_pretrain_1hop.sh
    ```
    This is used to obtain the initial node embeddings in the rgcn.

[5] Train the GCNTM-CK model with the scripts 'run_reuters_mr{}_path{}_num{}_{}hop.sh' acoording to 
    different settings. For instance, the script run_reuters_mr0.1_path50_num50_1hop.sh will train 
    a model with the following parameter settings:

        - H (hop number) -> 1
        - P (maximum number of pairs) -> 50
        - R (maximum number of nearest neighbors) -> 50
        - \lambda (manifold coefficient) -> 0.1

    ```
    bash run_reuters_mr0.1_path50_num50_1hop.sh
    ```
    We train 5 times for each different topic settings and report the average results.

[6] After finishing training all the model variants, use the script 'overall_results_reuters.sh' 
    to obtain the final evaluation results, which will contain three topic coherence scores 
    (c_v, c_npmi, c_uci) and one topic diversity score (td). The results will be saved in an xlsx 
    file. The results for the main setting (H=1, P=100, R=100, \lambda=0.1) in the paper can be 
    found in the folder 'final/'.

    The example log for some specifc topic setting of our model can be found in the folder 'models/' .

=========================================   End   =====================================================
