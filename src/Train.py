
import os
import shutil
import torch
from torch import nn
import numpy as np
from src.AMMN import PGD
from torch.autograd import Variable


def train_AMMN(model_name, train_data, val_data, test_data, model, epochs, initial_learning_rate, logpath, epsilon, alpha, t):
    logpath = open(logpath, 'a', encoding='utf-8')
    model_path = 'save/' + model_name + '/'
    if os.path.exists(model_path):  # 清除之前运行代码生成的模型
        shutil.rmtree(model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    best_valid_acc = 0
    best_valid_dir = ''
    pgd = PGD(model, 'trans_gcn', epsilon=epsilon, alpha=alpha)
    K = t  # 扰动次数
    criterion = nn.CrossEntropyLoss()
    print('Training...')
    for epoch in range(epochs + 1):
        p = epoch / epochs
        learning_rate = initial_learning_rate / pow((1 + 10 * p), 0.75)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # weight_decay=0.0001
        # 训练
        model.train()
        loss_d_vec = []
        loss_p_vec = []
        loss_final_vec = []
        train_acc_vec = []
        for data in train_data:
            users_tensor, items_tensor, modality_labels, link_labels = data
            if torch.cuda.is_available():
                users_tensor = users_tensor.cuda()
                items_tensor = items_tensor.cuda()
                modality_labels = modality_labels.cuda()
                link_labels = link_labels.cuda()
            # tensor不能反向传播，variable可以反向传播
            users_tensor = Variable(users_tensor)
            items_tensor = Variable(items_tensor)
            modality_labels = torch.squeeze(Variable(modality_labels))
            link_labels = torch.squeeze(Variable(link_labels))
            # 前向传播
            pred_links, pred_trans = model(users_tensor, items_tensor)
            # 计算损失
            loss_p = criterion(pred_links, link_labels)
            loss_d = criterion(torch.squeeze(pred_trans), modality_labels)
            loss_final = loss_p + loss_d

            loss_d_vec.append(loss_d.cpu().detach().numpy())
            loss_p_vec.append(loss_p.cpu().detach().numpy())
            loss_final_vec.append(loss_final.cpu().detach().numpy())

            _, argmax = torch.max(pred_links, 1)
            batch_acc = (argmax == link_labels).float().mean()
            train_acc_vec.append(batch_acc.item())

            # 正常训练 反向传播 得到正常的梯度
            loss_final.backward()
            # 对抗训练
            pgd.backup_grad()  # 备份模型全部梯度
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # first attack时备份模型全部参数
                # 在embedding上添加对抗扰动, 对抗部分的参数进行扰动更新, 包括备份也更新
                if t != K - 1:
                    optimizer.zero_grad()  # 模型全部梯度置0
                else:  # 最后一次扰动
                    pgd.restore_grad()  # 恢复模型全部梯度 即对抗前计算的梯度
                # 计算加入扰动后的新损失与新梯度
                _, adv_pred_d = model(users_tensor, items_tensor)
                loss_adv = criterion(torch.squeeze(adv_pred_d), modality_labels)
                loss_adv.backward()  # 反向传播，在原始正常的梯度基础上，逐次累加对抗训练K次扰动的梯度
            pgd.restore()   # 恢复对抗部分参数 原始参数 对抗部分新梯度
            # 至此 模型参数没有变化 对抗部分新梯度
            optimizer.step()  # 梯度下降算法，更新参数
            optimizer.zero_grad()  # 清空batch梯度,不影响下一batch计算

        loss_d = np.mean(loss_d_vec)
        loss_p = np.mean(loss_p_vec)
        loss_final = np.mean(loss_final_vec)
        train_acc = np.mean(train_acc_vec)
        write_infor = "epoch:[{}/{}], lr:{:.4f}, loss_final:{:.4f}, loss_p:{:.4f}, loss_d:{:.4f}".format(epoch, epochs, learning_rate, loss_final, loss_p, loss_d)
        print(write_infor)
        logpath.write('\n'+write_infor)

        # 验证
        model.eval()
        valid_acc, valid_pre, valid_auc = model.metrics_eval(val_data)
        write_infor = "Train acc:{:.4f}, Valid acc:{:.4f}, Valid pre:{:.4f}, Valid auc:{:.4f} ".format(train_acc, valid_acc, valid_pre, valid_auc)
        print(write_infor)
        logpath.write('\n'+write_infor)

        # 保存最好模型
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_dir = model_path + 'model' + str(epoch) + '.pkl'
            torch.save(model, best_valid_dir)

    # 加载最好模型 测试
    model.eval()
    print('Load best model ...')
    write_infor = 'Best model: ' + best_valid_dir
    print(write_infor)
    logpath.write('\n\n' + write_infor)
    model = torch.load(best_valid_dir)
    acc, pre, auc = model.metrics_eval(test_data)
    write_infor = "Test acc:{:.4f}, pre:{:.4f}, auc:{:.4f}".format(acc, pre, auc)
    print(write_infor)
    logpath.write('\n'+write_infor)
    logpath.close()


