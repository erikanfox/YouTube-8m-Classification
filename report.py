import numpy as np
import pandas as pd
import tensorflow as tf

"""
This files gives several functions to print model performances report:
1) print N performance metrics
2) print predicted labels
3) 
"""
##########################################################################################################
##########################################################################################################
"""
1) print N performance metrics
"""



##########################################################################################################
##########################################################################################################
"""
2) print predicted labels
"""

def get_label(label_index,top_n_labels,label_dict,get_names):
    def get_label_row(row):
        if type(row)!= np.ndarray:
            row= np.array(row)
        labels = label_dict.iloc[row[row<top_n_labels]].Name.values
        return labels
    if get_names:
      all_labels  = list(map(lambda row: get_label_row(row), label_index))
    else:
      all_labels  = list(map(np.array,label_index))
    return all_labels

def make_top_n_pred_df(pesudo_id,y_predproba,feat_labels,top_n_pred = 5,top_n_poplabels = 1000,get_names=True):
    label_dict = pd.read_csv("vocabulary.csv")
    top_labels_proba = np.fliplr(np.sort(y_predproba,axis=1))[:,:top_n_pred]
    top_labels_pred = np.fliplr(y_predproba.argsort(axis=1))[:,:top_n_pred]
    label_true = get_label(feat_labels,top_n_poplabels,label_dict,get_names)
    label_pred = get_label(top_labels_pred,top_n_poplabels,label_dict,get_names)
    # label_true = feat_labels
    # label_pred = top_labels_pred
    predict_df = pd.DataFrame({"pesudo_id":pesudo_id,
                "label_true":label_true,
                "label_pred":label_pred,
                "predict_proba":top_labels_proba.tolist()})
    return predict_df
    
