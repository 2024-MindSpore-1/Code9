
from torch.utils.data import DataLoader, Dataset
import numpy as np, pickle
import os,sys
from os.path import normpath,join
Base_DIR=normpath(join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0,Base_DIR)


def getGraphMap(data_root, num_path):
    relation_map = pickle.load(open(os.path.join(data_root, 'relation_map_path%d.pkl' % num_path), 'rb'))  
    unique_nodes_mapping = pickle.load(open(os.path.join(data_root, 'unique_nodes_mapping_path%d.pkl' % num_path), 'rb'))  
    concept_map = pickle.load(open(os.path.join(data_root, 'concept_map_path%d.pkl' % num_path), 'rb'))  

    return relation_map, concept_map, unique_nodes_mapping

def getConceptTriples(data_root, num_path): 
    ConceptTriples = pickle.load(open(os.path.join(data_root, 'all_doc_triples_path%d.pkl' % num_path), 'rb'))
    return ConceptTriples


def rawTriples2index(rawConceptTriples, map):
    relation_map = map[0]  
    concept_map = map[1]  
    unique_nodes_mapping = map[2] 

    conceptTriples = []
    for triple in rawConceptTriples:
        try:
            srcMap = concept_map[triple[0]]
            relMap = relation_map[triple[1]]
            distMap = concept_map[triple[2]]
            srcMap, distMap = unique_nodes_mapping[srcMap], unique_nodes_mapping[distMap]
        except:
            continue
        triple = [srcMap, relMap, distMap]
        conceptTriples.append(triple)
    # print(conceptTriples)
    return conceptTriples


class GraphDataset(Dataset): 
    def __init__(self, data_root, num_path):  

        self.rawDataset = getConceptTriples(data_root, num_path)
        self.maps = getGraphMap(data_root, num_path)

    def __len__(self): 
        return len(self.rawDataset)

    def __getitem__(self, index):  
        raw = self.rawDataset[index]  
        maps = self.maps
        Triples = rawTriples2index(raw, maps)

        return np.array(Triples)  


def getDataSet(data_root, num_path):
    graphdataset = GraphDataset(data_root, num_path)
    return graphdataset
