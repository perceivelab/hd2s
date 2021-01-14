# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:33:45 2020

@author: fproietto
"""
from torch.utils.data import DataLoader

# from gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()
        
    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
            if(len(batch[0])!=self.batch_size):
                batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch