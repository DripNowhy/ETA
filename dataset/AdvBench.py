import os

import pandas as pd
from dataset.base import BaseDataset


class AdvBench(BaseDataset):
    def __init__(self, data_root="YOUR_DATA_PATH"):
        super(AdvBench, self).__init__()
        self.ann_root = data_root

    def get_data(self):
        
        path = os.path.join(self.ann_root, "harmful_behaviors.csv")
        df = pd.read_csv(path)   

        data = df['goal'].tolist()

        return data