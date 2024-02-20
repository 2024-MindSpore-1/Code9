# ROOTPATH = '/YourHomePath/gntm-ck-code/'
ROOTPATH = '/home/cike/bingshan/gntm-ck-code/'
DATAPATH = ROOTPATH +'data/'
MODELPATH = ROOTPATH + 'models/'
EN_STOP_WORDS = DATAPATH +'EN_gensim_stopword.txt'

# 1 hop
Reuters_ADDR = DATAPATH +'reuters/'
NEWS20_ADDR = DATAPATH +'20news/'
BNC_ADDR = DATAPATH +'bnc/'
TMN_ADDR = DATAPATH +'tmn3/'
TMN_ALL_CONTENT_ADDR = DATAPATH + 'tmn3_all_content/'
WOS_ADDR = DATAPATH + 'wos11967/'

# 2 hop
Reuters_ADDR_2 = DATAPATH +'reuters_2hop/'
NEWS20_ADDR_2 = DATAPATH +'20news_2hop/'
BNC_ADDR_2 = DATAPATH +'bnc_2hop/'
TMN_ADDR_2 = DATAPATH +'tmn3_2hop/'
TMN_ALL_CONTENT_ADDR_2 = DATAPATH + 'tmn3_all_content_2hop/'
WOS_ADDR_2 = DATAPATH + 'wos11967_2hop/'

# 3 hop
Reuters_ADDR_3 = DATAPATH +'reuters_3hop/'
NEWS20_ADDR_3 = DATAPATH +'20news_3hop/'
BNC_ADDR_3 = DATAPATH +'bnc_3hop/'
TMN_ADDR_3 = DATAPATH +'tmn3_3hop/'
TMN_ALL_CONTENT_ADDR_3 = DATAPATH + 'tmn3_all_content_3hop/'
WOS_ADDR_3 = DATAPATH + 'wos11967_3hop/'

GLOVE_ADDR = DATAPATH +'word2vec/'

LABELED_DATASETS=['News20','WOS11967', 'TMN']
