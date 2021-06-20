
import torch
import torch.utils.data as data




    
class ComposeDatasetDict(data.Dataset):

    "TO take a dictionary of datasets and return a dataset which produces elements as a dictionalry "
    
    def __init__(self , data_loaders ):
        self.data_loaders = data_loaders
        
    def __getitem__(self, index):
        ret = {}
        for k in self.data_loaders:
            ret[k] = self.data_loaders[k].__getitem__(index)
        return ret 
   

def compose_dataset_dict(dataset_dict ):
    """TO take a dictionary of datasets and return a dataset which produces elements as a dictionalry 

    Args:
        dataset_dict ([type]): [description]
    """
    return  ComposeDatasetDict( dataset_dict )



