import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os

"""
This file stores useful functions to preprocess tfrecord,
including 1) get_records: getting all the files under a specified directory
          2) read_refords: parsing each recrods, getting id, labels,
                                  (n_frames, 1024) for RGB features,
                                    (n_frames, 128) fro audio features)
          3) avg_pooling: average pooling to aggregate input features from frames to video
          4) make_y: one hot encoding on labels for one-VS-all classifier
          5) preapre_logistic: using avg_pool, and make_y to prepare the input and output
                              for training baseline lgostic model
                              
"""

def get_records(directory, filetype):
    # tfrecords are stored in file folders, this function
    # get the names of all the record files under a directory
    r = re.compile("^%s.+\\.tfrecord$"%filetype)
    train_files =  os.listdir(directory)
    frames_records = sorted(list(filter(r.match, train_files)))
    frames_records = list(map(lambda orig_string: 
                            directory + orig_string, 
                        frames_records))
    return frames_records

def read_records(frames_records):
    # Parse tfrecords data
    feat_puesdoid = []
    feat_labels = []
    feat_rgb = []
    feat_audio = []
    for frames_record_i in frames_records:
        for example in tf.data.TFRecordDataset(frames_record_i):#.take(1000):        
            tf_seq_example = tf.train.SequenceExample()
            rf = tf_seq_example.ParseFromString(example.numpy())
            n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

            rgb_frame = []
            audio_frame = []
            # iterate through frames for audio and rgb features
            for i in range(n_frames):
                rgb_frame.append(tf.io.decode_raw(
                        tf_seq_example.feature_lists.feature_list["rgb"].feature[i].bytes_list.value[0],
                        tf.uint8))
                audio_frame.append(tf.io.decode_raw(
                        tf_seq_example.feature_lists.feature_list["audio"].feature[i].bytes_list.value[0],
                        tf.uint8))
            
            feat_rgb.append(rgb_frame)
            feat_audio.append(audio_frame)
            feat_puesdoid.append(tf_seq_example.context.feature["id"].bytes_list.value[0].decode(encoding="UTF-8"))
            feat_labels.append(tf_seq_example.context.feature["labels"].int64_list.value)

    return(feat_rgb,feat_audio,feat_puesdoid,feat_labels)



def avg_pooling(frame_data):
    # take avaerge across the frames for each video
    avg_rgb_by_vid = list(map(
        lambda frames: 
        np.array(frames).mean(axis=0),frame_data))

    X= np.array(avg_rgb_by_vid)
    return X


def make_y(labels,top_n_labels):
    # using a binary indicator for each label to suggest if a video contain each of the top 1000 labels
    # get y for one vs all classifier
    # dim: row = # of unique lable, col = number of video *20 frame per video
    unique_labels = np.arange(0,top_n_labels,1)
    if_label_by_label = [[] for i in range(len(unique_labels))]
    for i in range(len(unique_labels)):
        label = unique_labels[i]
        if_label_by_label[i] = list(map(lambda x: label in x, labels))
    if_label_by_label = np.array(if_label_by_label).T
    return if_label_by_label



def preprocess_for_logistic(feat_rgb,feat_audio,feat_labels, top_n_labels):
    # call avg_pooling( and make_y to get preprocessed inputs and outputs
    # for training average-pooling -based model
    
    X_rgb=avg_pooling(feat_rgb)
    X_rgb_tensor = tf.convert_to_tensor(X_rgb)
    X_audio =avg_pooling(feat_audio)
    X_audio_tensor = tf.convert_to_tensor(X_audio)

    y = make_y(feat_labels,top_n_labels)
    y_tensor = tf.convert_to_tensor(y)


    return(X_rgb_tensor, X_audio_tensor,y_tensor)
