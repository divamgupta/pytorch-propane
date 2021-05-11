import torch 
import numpy as np 
from six import string_types
from torch import optim
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F



def get_pt_loss_fn_from_str( loss_name ):
    
    loss_functions_dict = { 
        'mse': nn.MSELoss , 
        'mean_squared_error': nn.MSELoss , 
        'bce': nn.BCELoss ,
        'binary_cross_entropy' : nn.BCELoss  , 
        'smooth_l1_loss' : nn.SmoothL1Loss , 
        'l1' : nn.L1Loss  , 
        'cross_entropy' : nn.CrossEntropyLoss , 
        "nll_loss" :  nn.NLLLoss , 
    }
    
    if not loss_name in loss_functions_dict:
        raise ValueError("Loss string %s not found"%loss_name )
    
    return loss_functions_dict[ loss_name ]()
    




def get_loss_fn_from_pt_loss( pt_nn_loss ):
    """
    pt_nn_loss -> the pytorch nn loss modeule 
    """
    
    def loss_function( data_x=None , data_y=None , model_output=None ):
        return pt_nn_loss(  model_output , data_y )
    
    return loss_function


class Loss:
    def __init__( self , loss_fn=None  , key=None , loss_weight=1 , display_name=None ):
        
        '''
        loss_fn : string/ function / torch nn loss 
        key :  key / index of model output where the loss should be applied  
        '''
        
        if isinstance(loss_fn , string_types):
            loss_fn = get_pt_loss_fn_from_str( loss_fn )
            
        if isinstance( loss_fn , nn.modules.loss._Loss):
            loss_fn = get_loss_fn_from_pt_loss( loss_fn )
            
        self.loss_fn  = loss_fn
        self.loss_weight = loss_weight
        self.key = key 
        self.display_name = display_name 
        
    def __call__( self , model_output=None ,  data_y=None  , data_x=None  ):
        if self.key is None:
            return self.loss_weight*self.loss_fn( model_output=model_output ,data_y=data_y , data_x=data_x  )
        else:
            return self.loss_weight*self.loss_fn( model_output=model_output[self.key] ,data_y=data_y[self.key] , data_x=data_x  )
        



        