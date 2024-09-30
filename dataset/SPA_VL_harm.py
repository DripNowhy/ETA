from dataset.base import BaseDataset
from datasets import DatasetDict, load_dataset


class SPA_VL_harm(BaseDataset):
    def __init__(self, data_root="YOUR_DATA_PATH"):
        super(SPA_VL_harm, self).__init__()

        self.ann_root = data_root

    def get_data(self):
        eval_harm = load_dataset('parquet', data_files=f'{self.ann_root}/harm-00000-of-00001.parquet')

        dataset_dict = DatasetDict({'test': eval_harm['train']})

        return dataset_dict['test']