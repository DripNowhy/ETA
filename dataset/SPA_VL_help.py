from dataset.base import BaseDataset
from datasets import DatasetDict, load_dataset


class SPA_VL_help(BaseDataset):
    def __init__(self, data_root="/data/SPA_VL/"):
        super(SPA_VL_help, self).__init__()

        self.ann_root = data_root

    def get_data(self):
        eval_help = load_dataset('parquet', data_files=f'{self.ann_root}/help-00000-of-00001.parquet')

        dataset_dict = DatasetDict({'test': eval_help['train']})

        return dataset_dict['test']