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
parser.add_argument('-nc', '--num_of_clients', type=int, default=140, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=20, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=200, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
panduan = 6


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



for o in range(200):
    args = parser.parse_args()
    args = args.__dict__

    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger('200ceshi140%d.txt'%o)

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

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
    sum_acc_22 = []
    sum_acc_23 = []
    sum_acc_24 = []
    sum_acc_25 = []
    sum_acc_26 = []
    sum_acc_27 = []
    sum_acc_28 = []
    sum_acc_29 = []
    sum_acc_30 = []
    sum_acc_31 = []
    sum_acc_32 = []
    sum_acc_33 = []
    sum_acc_34 = []
    sum_acc_35 = []
    sum_acc_36 = []
    sum_acc_37 = []
    sum_acc_38 = []
    sum_acc_39 = []
    sum_acc_40 = []
    sum_acc_41 = []
    sum_acc_42 = []
    sum_acc_43 = []
    sum_acc_44 = []
    sum_acc_45 = []
    sum_acc_46 = []
    sum_acc_47 = []
    sum_acc_48 = []
    sum_acc_49 = []
    sum_acc_50 = []
    sum_acc_51 = []
    sum_acc_52 = []
    sum_acc_53 = []
    sum_acc_54 = []
    sum_acc_55 = []
    sum_acc_56 = []
    sum_acc_57 = []
    sum_acc_58 = []
    sum_acc_59 = []
    sum_acc_60 = []
    sum_acc_61 = []
    sum_acc_62 = []
    sum_acc_63 = []
    sum_acc_64 = []
    sum_acc_65 = []
    sum_acc_66 = []
    sum_acc_67 = []
    sum_acc_68 = []
    sum_acc_69 = []
    sum_acc_70 = []
    sum_acc_71 = []
    sum_acc_72 = []
    sum_acc_73 = []
    sum_acc_74 = []
    sum_acc_75 = []
    sum_acc_76 = []
    sum_acc_77 = []
    sum_acc_78 = []
    sum_acc_79 = []
    sum_acc_80 = []
    sum_acc_81 = []
    sum_acc_82 = []
    sum_acc_83 = []
    sum_acc_84 = []
    sum_acc_85 = []
    sum_acc_86 = []
    sum_acc_87 = []
    sum_acc_88 = []
    sum_acc_89 = []
    sum_acc_90 = []
    sum_acc_91 = []
    sum_acc_92 = []
    sum_acc_93 = []
    sum_acc_94 = []
    sum_acc_95 = []
    sum_acc_96 = []
    sum_acc_97 = []
    sum_acc_98 = []
    sum_acc_99 = []
    sum_acc_100 = []
    sum_acc_101 = []
    sum_acc_102 = []
    sum_acc_103 = []
    sum_acc_104 = []
    sum_acc_105 = []
    sum_acc_106 = []
    sum_acc_107 = []
    sum_acc_108 = []
    sum_acc_109 = []
    sum_acc_110 = []
    sum_acc_111 = []
    sum_acc_112 = []
    sum_acc_113 = []
    sum_acc_114 = []
    sum_acc_115 = []
    sum_acc_116 = []
    sum_acc_117 = []
    sum_acc_118 = []
    sum_acc_119 = []
    sum_acc_120 = []
    sum_acc_121 = []
    sum_acc_122 = []
    sum_acc_123 = []
    sum_acc_124 = []
    sum_acc_125 = []
    sum_acc_126 = []
    sum_acc_127 = []
    sum_acc_128 = []
    sum_acc_129 = []
    sum_acc_130 = []
    sum_acc_131 = []
    sum_acc_132 = []
    sum_acc_133 = []
    sum_acc_134 = []
    sum_acc_135 = []
    sum_acc_136 = []
    sum_acc_137 = []
    sum_acc_138 = []
    sum_acc_139 = []
    list_jiazong = [sum_acc_0, sum_acc_1, sum_acc_2, sum_acc_3, sum_acc_4, sum_acc_5, sum_acc_6, sum_acc_7, sum_acc_8,
                    sum_acc_9, sum_acc_10, sum_acc_11, sum_acc_12, sum_acc_13, sum_acc_14, sum_acc_15, sum_acc_16,
                    sum_acc_17, sum_acc_18, sum_acc_19, sum_acc_20, sum_acc_21, sum_acc_22, sum_acc_23, sum_acc_24,
                    sum_acc_25, sum_acc_26, sum_acc_27, sum_acc_28, sum_acc_29, sum_acc_30, sum_acc_31, sum_acc_32,
                    sum_acc_33, sum_acc_34, sum_acc_35, sum_acc_36, sum_acc_37, sum_acc_38, sum_acc_39, sum_acc_40,
                    sum_acc_41, sum_acc_42, sum_acc_43, sum_acc_44, sum_acc_45, sum_acc_46, sum_acc_47, sum_acc_48,
                    sum_acc_49, sum_acc_50, sum_acc_51, sum_acc_52, sum_acc_53, sum_acc_54, sum_acc_55, sum_acc_56,
                    sum_acc_57, sum_acc_58, sum_acc_59, sum_acc_60, sum_acc_61, sum_acc_62, sum_acc_63, sum_acc_64,
                    sum_acc_65, sum_acc_66, sum_acc_67, sum_acc_68, sum_acc_69, sum_acc_70, sum_acc_71, sum_acc_72,
                    sum_acc_73, sum_acc_74, sum_acc_75, sum_acc_76, sum_acc_77, sum_acc_78, sum_acc_79, sum_acc_80,
                    sum_acc_81, sum_acc_82, sum_acc_83, sum_acc_84, sum_acc_85, sum_acc_86, sum_acc_87, sum_acc_88,
                    sum_acc_89, sum_acc_90, sum_acc_91, sum_acc_92, sum_acc_93, sum_acc_94, sum_acc_95, sum_acc_96,
                    sum_acc_97, sum_acc_98, sum_acc_99, sum_acc_100, sum_acc_101, sum_acc_102, sum_acc_103, sum_acc_104,
                    sum_acc_105, sum_acc_106, sum_acc_107, sum_acc_108, sum_acc_109, sum_acc_110, sum_acc_111, sum_acc_112,
                    sum_acc_113, sum_acc_114, sum_acc_115, sum_acc_116, sum_acc_117, sum_acc_118, sum_acc_119, sum_acc_120,
                    sum_acc_121, sum_acc_122, sum_acc_123, sum_acc_124, sum_acc_125, sum_acc_126, sum_acc_127, sum_acc_128,
                    sum_acc_129, sum_acc_130, sum_acc_131, sum_acc_132, sum_acc_133, sum_acc_134, sum_acc_135, sum_acc_136, sum_acc_137, sum_acc_138, sum_acc_139]
    for i in range(140):
        sum_acc_jiedian.append(0)

    global_accuracy = 0
    global_accuracy_list = []

    for i in range(args['num_comm']):
        if i != 200:
            print("communicate round {}".format(i+1))

            order = []
            for k in range(140):
                order.append(k)
            clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
            j = 0

            sum_parameters = None
            client_num = 0
            for client in tqdm(clients_in_comm):
                print(client)
                local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                             loss_func, opti, global_parameters)
                net.load_state_dict(local_parameters, strict=True)
                local_accu = 0
                local_num = 0

                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    local_accu += (preds == label).float().mean()
                    local_num += 1
                print('local_accuracy: {}'.format(local_accu / local_num))
                local_accuracy = float(format(float(local_accu / local_num)))


                if local_accuracy != global_accuracy:
                    pass
                else:
                    print('第%d个节点可能为恶意节点' % client_num)
                    print('开始检测：')
                    jishu = 0
                    jiance = 0

                    for var in local_parameters:
                        if torch.equal(local_parameters[var], global_parameters[var]) == True:
                            jiance += 1
                        jishu += 1
                    if jiance == jishu:
                        print('第%d个节点为恶意节点' % client_num)
                    else:
                        print('第%d个节点不是恶意节点' % client_num)



                sum_acc_jiedian[j] = sum_acc_jiedian[j] + local_accuracy
                list_jiazong[j % 140].append(local_accuracy)
                j += 1

                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] + local_parameters[var]
                client_num += 1

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
            global_accuracy = float(format(float(sum_accu / num)))
            global_accuracy_list.append(global_accuracy)

            if (i + 1) % args['save_freq'] == 0:
                torch.save(net, os.path.join(args['save_path'],
                                             '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                    i, args['epoch'],
                                                                                                    args['batchsize'],
                                                                                                    args['learning_rate'],
                                                                                                    args['num_of_clients'],
                                                                                                    args['cfraction'])))
            print(sum_acc_jiedian)
            if i == 5:
                sum_avg_acc_sort_1 = []
                sum_avg_acc_sort_2 = []
                sum_avg_acc_sort_3 = []
                for k in range(140):
                    if k < 60:
                        sum_avg_acc_sort_1.append(sum_acc_jiedian[k])
                    elif k < 120:
                        sum_avg_acc_sort_2.append(sum_acc_jiedian[k])
                    else:
                        sum_avg_acc_sort_3.append(sum_acc_jiedian[k])


                avg_1 = sum(sum_avg_acc_sort_1) / 60
                var1 = (np.var(sum_avg_acc_sort_1)) ** 0.5
                low_1 = avg_1 - 1 * var1
                print('var1', var1)
                print('low_1', low_1)
                avg_2 = sum(sum_avg_acc_sort_2) / 60
                var2 = (np.var(sum_avg_acc_sort_2)) ** 0.5
                low_2 = avg_2 - 3 * var2
                print('low_2', low_2)
                print('var2', var2)
                avg_3 = sum(sum_avg_acc_sort_3) / 20
                var3 = (np.var(sum_avg_acc_sort_3)) ** 0.5
                low_3 = avg_3 - 3 * var3
                print('var3', var3)
                print('low_3', low_3)

                for kb in range(60):
                    if sum_avg_acc_sort_1[kb] < low_1:
                        print('第%d个节点被认为是恶意节点或未达到合约精度' % kb)
                    else:
                        print('第%d个节点被认为是正常节点' % kb)
                for kb in range(60):
                    if sum_avg_acc_sort_2[kb] < low_2:
                        print('第%d个节点被认为是恶意节点或未达到合约精度' % (60 + kb))
                    else:
                        print('第%d个节点被认为是正常节点' % (60 + kb))
                for kb in range(20):
                    if sum_avg_acc_sort_3[kb] < low_3:
                        print('第%d个节点被认为是恶意节点或未达到合约精度' % (120 + kb))
                    else:
                        print('第%d个节点被认为是正常节点' % (120 + kb))
                sum_avg_acc_sort_1.clear()
                sum_avg_acc_sort_2.clear()
                sum_avg_acc_sort_3.clear()



        else:
            sum_avg_acc = []
            sum_avg_acc_sort = []
            sum_avg_acc_sort_1 = []
            sum_avg_acc_sort_2 = []
            sum_avg_acc_sort_3 = []
            sum_avg_acc_sort_4 = []
            for k in list_jiazong:
                gd_acc = sum(k) / (panduan + 1)
                sum_avg_acc.append(gd_acc)
            print('sum_avg_acc', sum_avg_acc)
            for k in range(140):
                if k < 60:
                    sum_avg_acc_sort_1.append(sum_avg_acc[k])
                elif k < 120:
                    sum_avg_acc_sort_2.append(sum_avg_acc[k])
                else:
                    sum_avg_acc_sort_3.append(sum_avg_acc[k])
            avg_1 = sum(sum_avg_acc_sort_1) / 60

            var1 = np.var(sum_avg_acc_sort_1)
            low_1 = avg_1 - 5 * var1
            print('var1', var1)
            print('low_1', low_1)
            avg_2 = sum(sum_avg_acc_sort_2) / 60
            var2 = np.var(sum_avg_acc_sort_2)
            low_2 = avg_2 - 5 * var2
            print('low_2', low_2)
            print('var2', var2)
            avg_3 = sum(sum_avg_acc_sort_3) / 20
            var3 = np.var(sum_avg_acc_sort_3)
            low_3 = avg_3 - 5 * var3
            print('var3', var3)
            print('low_3', low_3)

            for kb in range(60):
                if sum_avg_acc_sort_1[kb] < low_1:
                    print('第%d个节点被认为是恶意节点或未达到合约精度' % kb)
                else:
                    print('第%d个节点被认为是正常节点' % kb)
            for kb in range(60):
                if sum_avg_acc_sort_2[kb] < low_2:
                    print('第%d个节点被认为是恶意节点或未达到合约精度' % (60 + kb))
                else:
                    print('第%d个节点被认为是正常节点' % (60 + kb))
            for kb in range(20):
                if sum_avg_acc_sort_3[kb] < low_3:
                    print('第%d个节点被认为是恶意节点或未达到合约精度' % (120 + kb))
                else:
                    print('第%d个节点被认为是正常节点' % (120 + kb))

            sum_avg_acc.clear()
    print(global_accuracy_list)
