
import os
import math
import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader


def split_by_xlabel(data):
    group = data.groupby('x_label')
    inters = []
    for x_label, infors in group:
        inters.append(infors)
    return inters[0], inters[1], inters[2]

def trainset_load(train_inters, batch_size):
    users = torch.LongTensor(np.array(train_inters['userID']))
    pos_items = torch.LongTensor(np.array(train_inters['itemID']))
    neg_items = torch.LongTensor(np.array(train_inters['neg_itemID']))
    pos_modality = torch.LongTensor(np.array(train_inters['pos_modality']))
    neg_modality = torch.LongTensor(np.array(train_inters['neg_modality']))
    pos_labels = torch.LongTensor([1] * pos_items.shape[0])
    neg_labels = torch.LongTensor([0] * neg_items.shape[0])
    users_tensor = torch.cat((torch.unsqueeze(users, 1), torch.unsqueeze(users, 1)), dim=0)
    items_tensor = torch.cat((torch.unsqueeze(pos_items, 1), torch.unsqueeze(neg_items, 1)), dim=0)
    modality_labels = torch.cat((torch.unsqueeze(pos_modality, 1), torch.unsqueeze(neg_modality, 1)), dim=0)
    link_labels = torch.cat((torch.unsqueeze(pos_labels, 1), torch.unsqueeze(neg_labels, 1)), dim=0)
    train_set = TensorDataset(users_tensor, items_tensor, modality_labels, link_labels)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # 打乱
    return train_loader

def eval_data_load(inters, batch_size):
    users = torch.LongTensor(np.array(inters['userID']))
    pos_items = torch.LongTensor(np.array(inters['itemID']))
    neg_items = torch.LongTensor(np.array(inters['neg_itemID']))
    pos_labels = torch.LongTensor([1] * pos_items.shape[0])
    neg_labels = torch.LongTensor([0] * neg_items.shape[0])
    users_tensor = torch.cat((torch.unsqueeze(users, 1), torch.unsqueeze(users, 1)), dim=0)
    items_tensor = torch.cat((torch.unsqueeze(pos_items, 1), torch.unsqueeze(neg_items, 1)), dim=0)
    link_labels = torch.cat((torch.unsqueeze(pos_labels, 1), torch.unsqueeze(neg_labels, 1)), dim=0)
    data_set = TensorDataset(users_tensor, items_tensor, link_labels)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader

def uncertain_missing(missing_case_count, proportion):
    # 未缺失 1-proportion [missing_case_count-1]
    # 随机不确定缺失 proportion [0, missing_case_count-2]
    probability = []
    non_missing = 1-proportion
    for i in range(missing_case_count-2):  # [0, missing_case_count-3]
        probability.append(random.uniform(0, proportion))
        proportion -= probability[i]
    probability.append(proportion)  # [missing_case_count-2]
    probability.append(non_missing)  # [missing_case_count-1]
    encode = np.random.choice([i for i in range(missing_case_count)], p=probability)
    return encode

def DecodeCase(encode, modal_count):
    is_exist = [0] * modal_count
    for i in range(modal_count):
        m = math.pow(2, modal_count-1-i)
        if encode//m == 1:
            is_exist[i] = 1
        encode %= m
    return is_exist

def modal_data(dir_str, userIDlen, itemIDlen, proportion, save_path):
    # 模态数量 modal_count -> 不确定缺失情况数量 missing_case_count
    feat_files = []
    for filename in os.listdir(dir_str):
        if 'feat' in filename:
            feat_files.append(dir_str + filename)
    modal_count = len(feat_files)
    missing_case_count = int(math.pow(2, modal_count))
    feats = []
    feats_dim = []
    # 模拟不确定缺失 得到gcn需要的x
    for i in range(modal_count):
        feats.append(np.load(feat_files[i], allow_pickle=True))
        feats_dim.append(feats[i].shape[1])
    origin_dims = sum(feats_dim)
    # user节点
    user_x = np.zeros((userIDlen, origin_dims))
    # item节点
    item_modalitys = {}
    item_x = []
    missing = [[], []]
    for item in range(itemIDlen):
        encode = uncertain_missing(missing_case_count, proportion)
        item_modalitys[item + userIDlen] = encode
        is_exist = DecodeCase(encode, modal_count)
        missing_feat = []
        for i in range(modal_count):
            if is_exist[i]:  # 对应模态存在
                missing_feat += feats[i][item].tolist()
                missing[i].append(feats[i][item].tolist())
            else:  # 对应模态缺失
                missing_feat += [0] * feats_dim[i]
                missing[i].append([0] * feats_dim[i])
        item_x.append(missing_feat)
    # print('Item Feat:')
    # for i in range(modal_count):
    #     missing[i] = np.array(missing[i])
    #     print(missing[i].shape)
    #     np.save(save_path + 'feat' + str(i) + '.npy', missing[i])
    x = np.concatenate((user_x, np.array(item_x)), axis=0)
    x = torch.FloatTensor(x)
    print('Uncertain missing simulated completion ')
    return modal_count, missing_case_count, origin_dims, x, item_modalitys

def gcndata_load(train_inters, nodelist, gcn_x):
    # 用train_pos_edge构图 得到gcn需要的edge_index
    train_pos_edge = np.array(train_inters[['userID','itemID']]).tolist()
    g = nx.Graph(train_pos_edge)  # 交互关系转换为图
    g.add_nodes_from(nodelist)  # 有的节点只在验证测试阶段出现 训练阶段是未存在交互关系的孤立节点
    adj = nx.to_scipy_sparse_matrix(g, nodelist=nodelist, dtype=int, format='coo')  # 生成图的邻接矩阵的稀疏矩阵
    edge_index = torch.LongTensor(np.vstack((adj.row, adj.col)))  # 得到gcn需要的coo形式的edge_index
    gcn_data = Data(x=gcn_x, edge_index=edge_index)
    return gcn_data

def obtain_neg_sample(inter, all_items):
    group = inter.groupby('userID')
    neg_samples = []
    for user, user_items in group:
        pos_list = user_items['itemID'].tolist()
        neg_samples += random.sample(list(filter(lambda x: x not in pos_list, all_items)), len(pos_list))
    inter['neg_itemID'] = neg_samples
    return inter

def load_data(dataset, batch_size, proportion):
    dir_str = './data/' + dataset + '/'
    interfile = dir_str + dataset + '.inter'
    inter = pd.read_csv(interfile, sep='\t')
    userID = inter['userID'].drop_duplicates().to_list()
    itemID = inter['itemID'].drop_duplicates().to_list()
    userIDlen = len(userID)
    itemIDlen = len(itemID)
    print('***************************')
    print('Dataset:', dataset)  # baby
    print('Numbers of users:', userIDlen)  # 19445
    print('Numbers of items:', itemIDlen)  # 7050
    print('Numbers of inters:', inter.shape[0])  # 160792
    print('***************************')

    if not os.path.exists('save/'):
        os.mkdir('save/')
    save_path = 'save/' + dataset + '_p_' + str(proportion) + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    inter['itemID'] += userIDlen  # id低范围为user,高范围为item
    nodelist = [i for i in range(userIDlen + itemIDlen)]  # 全部节点
    all_items = nodelist[userIDlen:]  # item节点高范围
    # 负采样
    inter = obtain_neg_sample(inter, all_items)
    # 模态数量->不确定缺失情况数量 模拟不确定缺失 得到gcn需要的x
    modal_count, missing_case_count, origin_dims, gcn_x, item_modalitys = modal_data(dir_str, userIDlen, itemIDlen, proportion, save_path)
    inter['pos_modality'] = torch.LongTensor(inter['itemID'].map(item_modalitys))
    inter['neg_modality'] = torch.LongTensor(inter['neg_itemID'].map(item_modalitys))
    # 8:1:1 on the interaction history of each user并负采样
    train_inters, valid_inters, test_inters = split_by_xlabel(inter)
    print('Train triplet counter:', train_inters.shape[0])
    print('Valid triplet counter:', valid_inters.shape[0])
    print('Test triplet counter:', test_inters.shape[0])
    # valid_inters[['userID', 'itemID', 'neg_itemID']].to_csv(save_path + 'valid_inters', index=False)
    # test_inters[['userID', 'itemID', 'neg_itemID']].to_csv(save_path + 'test_inters', index=False)
    # 用train_pos_edge构图 得到gcn需要的edge_index
    gcn_data = gcndata_load(inter, nodelist, gcn_x)
    print('Network construction completed')
    # train_pos_edge 以及 它对应的 user_pos_items_dict
    # val_pos_items 以及 test_pos_items
    train_data = trainset_load(train_inters, batch_size)
    valid_data = eval_data_load(valid_inters, batch_size)
    test_data = eval_data_load(test_inters, batch_size)
    print('Data load completed')
    print('***************************')
    return userIDlen, itemIDlen, missing_case_count, origin_dims, gcn_data, train_data, valid_data, test_data
