import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import math
import random

class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        # self.random_size = math.pow(random_size, 1.3)

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()
        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label
        index = [i for i in range(len(train_data))]
        random.shuffle(index)
        train_data = train_data[index]
        train_label = train_label[index]


        for i in range(self.num_of_clients):

            if i < 8:
                shard_size = 600
                data_shards = train_data[i*shard_size: i*shard_size + shard_size]
                label_shards = train_label[i*shard_size: i*shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)
                # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('???', i, '?????????:', shard_size)
            elif 7 < i < 15:
                shard_size = 300
                j = i - 8
                data_shards = train_data[4800 + j * shard_size: 4800 + j * shard_size + shard_size]
                label_shards = train_label[4800 + j * shard_size: 4800 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)
                # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('???', i, '?????????:', shard_size)
            elif 14 < i < 20:
                shard_size = 100
                j = i - 15
                data_shards = train_data[6900 + j * shard_size: 6900 + j * shard_size + shard_size]
                label_shards = train_label[6900 + j * shard_size: 6900 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)

                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('???', i, '?????????:', shard_size)
            else:
                shard_size = 800
                j = i - 20
                data_shards = train_data[7400 + j * shard_size: 7400 + j * shard_size + shard_size]
                label_shards = train_label[7400 + j * shard_size: 7400 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)

                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('???', i, '?????????:', shard_size)


if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


