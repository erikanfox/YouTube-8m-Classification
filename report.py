import eval_util as eval
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import pandas as pd


##########################################################################################################
def report_performance(label_proba,label_true,verbose = True,thresh_step = 0.01,thresh=None):
    if type(label_true)!=np.ndarray:
        label_true = label_true.numpy()
        
    #three metrics from the paper
    gAP= eval.calculate_gap(label_proba,label_true)
    PERR = eval.calculate_precision_at_equal_recall_rate(label_proba,label_true)
    HIT1= eval.calculate_hit_at_one(label_proba,label_true)


    # weighted average f1 score
    n_labels= label_true.shape[1]
    if thresh== None:
        # if a specific threshold is not give, test all possible and give the optimal
        threshold_list = np.arange(0,1,thresh_step)
        threshold_list=threshold_list[1:]
        F1 = np.zeros(len(threshold_list))
        for i in range(len(threshold_list)):
            metric = tfa.metrics.F1Score(num_classes=n_labels,average="weighted",threshold=threshold_list[i])
            metric.update_state(label_true, label_proba)
            result = metric.result().numpy()
            F1[i]= result

        thresh_optimal = threshold_list[np.argmax(F1)]
        F1_optimal = np.max(F1)

        if verbose:
            print("gAP = %.4f, PERR = %.4f, HIT1 = %.4f"%(gAP,PERR, HIT1))
            print("Optimal weigthed F1 score %.4f when treshold = %.4f"%(F1_optimal,thresh_optimal))
        return gAP,PERR, HIT1,F1_optimal,thresh_optimal
    
    else:
        # if given a specific threshold, use it for F1 score
        metric = tfa.metrics.F1Score(num_classes=n_labels,average="weighted",threshold=thresh)
        metric.update_state(label_true, label_proba)
        F1 = metric.result().numpy()
        if verbose:
            print("gAP = %.4f, PERR = %.4f, HIT1 = %.4f"%(gAP,PERR, HIT1))
            print("Weigthed F1 score %.4f when treshold = %.4f"%(F1,thresh))
        return gAP,PERR, HIT1,F1
        
        

##########################################################################################################

# label_dict = pd.read_csv("vocabulary.csv")


def make_top_n_pred_df(pseudo_id,y_predproba,feat_labels,top_n_pred = 5,top_n_poplabels = 1000,get_names=False):
    top_labels_proba = np.fliplr(np.sort(y_predproba,axis=1))[:,:top_n_pred]
    top_labels_pred = np.fliplr(y_predproba.argsort(axis=1))[:,:top_n_pred]

    label_true = feat_labels
    label_pred = top_labels_pred
    predict_df = pd.DataFrame({"pseudo_id":pseudo_id,
                "label_true":label_true,
                "label_pred":label_pred,
                "predict_proba":top_labels_proba.tolist(),})
    return predict_df
    

    
