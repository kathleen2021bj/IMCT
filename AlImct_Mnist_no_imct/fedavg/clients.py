import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import random


class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            sum_accu = 0
            num = 0
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                predss = torch.argmax(preds, dim=1)
                sum_accu += (predss == label).float().mean()
                num += 1
                opti.step()
                opti.zero_grad()
            print('第', epoch + 1, '次,local_train', 'accuracy: {}'.format(sum_accu / num))


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
            if i < 40:
                shard_size = 500
                data_shards = train_data[i*shard_size: i*shard_size + shard_size]
                label_shards = train_label[i*shard_size: i*shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)
                # someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev,)
                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))
            elif 39 < i < 60:
                shard_size = 100
                j = i - 40
                data_shards = train_data[20000 + j * shard_size: 20000 + j * shard_size + shard_size]
                label_shards = train_label[20000 + j * shard_size: 20000 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)

                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))
            elif 59 < i < 120:
                shard_size = 300
                j = i - 60
                data_shards = train_data[22000 + j * shard_size: 22000 + j * shard_size + shard_size]
                label_shards = train_label[22000 + j * shard_size: 22000 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)

                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))
            else:
                shard_size = 1000
                j = i - 120
                data_shards = train_data[40000 + j * shard_size: 40000 + j * shard_size + shard_size]
                label_shards = train_label[40000 + j * shard_size: 40000 + j * shard_size + shard_size]
                local_data, local_label = np.vstack(data_shards), np.vstack(label_shards)
                local_label = np.argmax(local_label, axis=1)

                someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
                self.clients_set['client{}'.format(i)] = someone
                print('第', i, '个节点:', len(data_shards))

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


