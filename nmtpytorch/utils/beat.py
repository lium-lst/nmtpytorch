# -*- coding: utf-8 -*-

"""
Fantastic tool about...

"""

def beat_separate_train_valid(data_dict):
    data_dict_train = {}
    data_dict_train['src'] = []
    data_dict_train['trg'] = []
    data_dict_dev = {}
    data_dict_dev['src'] = []
    data_dict_dev['trg'] = []
    nb_doc_dev = 0
    nb_sent_dev = 0
    nb_doc_train = 0
    nb_sent_train = 0
    for fid in data_dict:
        if data_dict[fid]['file_info']['in_dev']:
            nb_doc_dev += 1
            #print("IN DEV #{}".format(nb_dev))
            for s in data_dict[fid]['source']:
                nb_sent_dev += 1
                data_dict_dev['src'].append(s)
            for s in data_dict[fid]['target']:
                data_dict_dev['trg'].append(s)
        else:
            nb_doc_train += 1
            #print("IN TRAIN #{}".format(nb_train))
            for s in data_dict[fid]['source']:
                nb_sent_train += 1
                data_dict_train['src'].append(s)
            for s in data_dict[fid]['target']:
                data_dict_train['trg'].append(s)

    #print("################################# DATA STATS: train len = {} doc and {} sents".format(nb_doc_train, nb_sent_train))
    #print("################################# DATA STATS: valid len = {} docs = {} sents".format(nb_doc_dev, nb_sent_dev))

    return data_dict_train, data_dict_dev

