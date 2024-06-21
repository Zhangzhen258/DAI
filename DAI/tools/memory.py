import random
import copy
import torch
import torch.nn.functional as F
import numpy as np

class FIFO():
    def __init__(self, capacity):
        self.data = [[], []]
        self.capacity = capacity
        pass

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert (len(instance) == 2)

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass

class Reservoir(): # Time uniform

    def __init__(self, capacity):
        super(Reservoir, self).__init__(capacity)
        self.data = [[], [], []]
        self.capacity = capacity
        self.counter = 0

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])


    def add_instance(self, instance):
        assert (len(instance) == 3)
        is_add = True
        self.counter+=1

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance()

        if is_add:
            for i, dim in enumerate(self.data):
                dim.append(instance[i])


    def remove_instance(self):


        m = self.get_occupancy()
        n = self.counter
        u = random.uniform(0, 1)
        if u <= m / n:
            tgt_idx = random.randrange(0, m)  # target index to remove
            for dim in self.data:
                dim.pop(tgt_idx)
        else:
            return False
        return True

class PBRS():

    def __init__(self, capacity, num_class):
        self.data = [[[], []] for _ in range(num_class)] #feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * num_class
        self.marker = [''] * num_class
        self.capacity = capacity
        self.num_class = num_class
        pass

    def get_memory(self):
        data = self.data
        tmp_data = [[], []]
        for data_per_cls in data:
            feats, cls = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def add_instance(self, instance):
        assert (len(instance) == 2)
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if cls not in largest_indices: #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(0, len(self.data[largest][0]))  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:# replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(0, len(self.data[cls][0]))  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True