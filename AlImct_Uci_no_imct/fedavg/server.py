import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
import sys

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=22, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=300, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('9000train3.txt')

if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    sum_acc_jiedian = []
    sum_acc_0 = []
    sum_acc_1 = []
    sum_acc_2 = []
    sum_acc_3 = []
    sum_acc_4 = []
    sum_acc_5 = []
    sum_acc_6 = []
    sum_acc_7 = []
    sum_acc_8 = []
    sum_acc_9 = []
    sum_acc_10 = []
    sum_acc_11 = []
    sum_acc_12 = []
    sum_acc_13 = []
    sum_acc_14 = []
    sum_acc_15 = []
    sum_acc_16 = []
    sum_acc_17 = []
    sum_acc_18 = []
    sum_acc_19 = []
    sum_acc_20 = []
    sum_acc_21 = []
    list_jiazong = [sum_acc_0, sum_acc_1, sum_acc_2, sum_acc_3, sum_acc_4, sum_acc_5, sum_acc_6, sum_acc_7, sum_acc_8,
    sum_acc_9, sum_acc_10, sum_acc_11, sum_acc_12, sum_acc_13, sum_acc_14, sum_acc_15, sum_acc_16, sum_acc_17, sum_acc_18,
    sum_acc_19, sum_acc_20, sum_acc_21]

    for i in range(22):
        sum_acc_jiedian.append(0)


    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        # order = np.random.permutation(args['num_of_clients'])
        order = []
        for p in range(num_in_comm):
            order.append(p)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        j = 0
        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            net.load_state_dict(local_parameters, strict=True)
            local_sum_accu = 0
            local_num = 0
            for data, label in testDataLoader:
                data, label = data.to(dev), label.to(dev)
                preds = net(data)
                preds = torch.argmax(preds, dim=1)
                local_sum_accu += (preds == label).float().mean()
                local_num += 1
            print('local_accuracy: {}'.format(local_sum_accu / local_num))
            local_accuracy = float(local_sum_accu / local_num)
            sum_acc_jiedian[j] = sum_acc_jiedian[j] + local_accuracy
            list_jiazong[j % 22].append(local_accuracy)
            j += 1



            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():

                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('global_accuracy: {}'.format(sum_accu / num))

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))
        print(sum_acc_jiedian)
        print(list_jiazong)

