import os
import torch
import argparse
import datetime

from src.Load import load_data
from src import AMMN
from src.Train import train_AMMN

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AMMN', help='name of model')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of dataset')
    parser.add_argument('--proportion', '-p', type=float, default=0.5, help='uncertain missing proportion')
    parser.add_argument('--learning_rate', '-l', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='number of iterations')
    parser.add_argument('--batch_size', '-b', type=int, default=512, help='batch size of train data')
    parser.add_argument('--dim_node', '-dim', type=int, default=32, help='dims of embedding')
    parser.add_argument('--epsilon', '-epsilon', type=int, default=8, help='epsilon of PGD')
    parser.add_argument('--alpha', '-alpha', type=int, default=4, help='alpha of PGD')
    parser.add_argument('--t', '-t', type=int, default=3, help='number of disturbances of PGD')
    parser.add_argument('--repeats', '-r', type=int, default=1, help='number of repeats')
    parser.add_argument('--logpath', '-f', type=str, default='log', help='save path of log')

    args = parser.parse_args()
    if not os.path.exists(args.logpath):
        os.mkdir(args.logpath)

    print('The program starts running.')
    for repeat in range(args.repeats):
        print('***************************')
        begin = datetime.datetime.now()
        print('Start time ', begin)
        time = str(begin.year) + '-' + str(begin.month) + '-' + str(begin.day) + '-' + str(begin.hour) + '-' + str(
            begin.minute) + '-' + str(begin.second)
        args.logpath = 'log/' + args.dataset + '-' + time + '.txt'
        print(args)

        logpath = open(args.logpath, 'a', encoding='utf-8')
        write_infor = "\nproportion:{}, lr:{}, epoch:{}, batch:{}, dim:{}, epsilon:{}, alpha:{}, t:{}".format(args.proportion, args.learning_rate, args.epochs, args.batch_size, args.dim_node, args.epsilon, args.alpha, args.t)+'\n'
        logpath.write(write_infor)
        logpath.close()

        # load data
        num_user, num_item, missing_case_count, origin_dims, gcn_data, train_data, val_data, test_data = load_data(
            args.dataset, args.batch_size, args.proportion)
        # load model
        my_model = AMMN.AMMN_Net(origin_dims, args.dim_node, missing_case_count, num_user, num_item, gcn_data)
        # run model
        if torch.cuda.is_available():
            my_model = my_model.cuda()
            gcn_data.x = gcn_data.x.cuda()
            gcn_data.edge_index = gcn_data.edge_index.cuda()
        train_AMMN(args.model, train_data, val_data, test_data, my_model, args.epochs, args.learning_rate, args.logpath, args.epsilon, args.alpha, args.t)

        # time
        end = datetime.datetime.now()
        print('End time ', end)
        print('Run time ', end-begin)
