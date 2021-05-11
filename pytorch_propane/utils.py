
import torch 
import numpy as np 
from six import string_types
from torch import optim
import inspect 
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm 




        
    
def get_vars( data , cuda=False  , numpy=False ):
    
#     list( map( lambda x :Variable(torch.FloatTensor(x.float() )).cuda() , imgs   ))
    
    
    if type( data ) is tuple:
        return tuple([ get_vars(d , cuda=cuda , numpy=numpy) for d in data  ])
    elif type( data ) is list:
        return list([ get_vars(d , cuda=cuda , numpy=numpy) for d in data  ])
    elif type( data ) is dict:
        return { k:get_vars(data[k] , cuda=cuda , numpy=numpy) for k in data }
    else:
        if numpy:
            data = torch.from_numpy(data)
        r =  Variable( data )
        if cuda:
            r = r.cuda()
        return r
    
def get_np_arrs( data ):
    if type( data ) is tuple:
        return tuple([ get_np_arrs(d  ) for d in data  ])
    elif type( data ) is list:
        return list([ get_np_arrs(d  ) for d in data  ])
    elif type( data ) is dict:
        return { k:get_np_arrs(data[k]  ) for k in data }
    else:
        return data.cpu().detach().numpy()
    
    


class ProgressBar(tqdm):
    def __init__( self , iterator ):
        super(ProgressBar, self).__init__(iterator)
        self.vals_history_dict = {}
        
    def add( self , vals_dict ):
        for k in vals_dict:
            if not k in self.vals_history_dict:
                self.vals_history_dict[k] = []
            self.vals_history_dict[k].append( vals_dict[k])
        
        bar_str = ""
        for k in self.vals_history_dict:
            bar_str += k+":"+ "%.3f"%(np.mean(self.vals_history_dict[k])) + " "
        
        self.set_description(bar_str )
        
    


    