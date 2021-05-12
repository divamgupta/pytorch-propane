

import torch 
import numpy as np 
from torch import optim
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F




class InferenceCallback:
    def __init__( self):
        pass 
    
    def on_start( self , *args , **kargs ):
        pass 
    
    def on_end(self , *args , **kargs ):
        pass
    
    def __call__( self , data_x=None , data_y=None , model_output=None  ):
        pass 
    

    
    
    
class Callback:
    def __init__(self , callback_fn=None  , event=None , frequency=1 ):
        
        self.on_train_batch_end_fn = None
        self.on_epoch_end_fn = None
        self.on_train_end_fn = None
        self.frequency = frequency 
        
        if not callback_fn is None:
            if event == "on_train_batch_end":
                self.on_train_batch_end_fn  = callback_fn 
            elif event == "on_epoch_end":
                self.on_epoch_end_fn  = callback_fn 
            elif event == "on_train_end":
                self.on_train_end_fn  = callback_fn 
            else:
                raise ValueError("invalid event %s"%event )
            
        self.n_done = 0
    
    def on_train_end(self , logs=None):
        self.n_done += 1 
        
        if self.n_done % self.frequency != 0:
            return 
        
        if not self.on_train_end_fn is None:
            self.on_train_end_fn( logs=logs )
        
    def on_epoch_end(self , epoch , logs=None):
        self.n_done += 1 
        
        if self.n_done % self.frequency != 0:
            return 
        
        if not self.on_epoch_end_fn is None:
            self.on_epoch_end_fn( epoch=epoch , logs=logs )
        
    def on_train_batch_end( self , batch , logs=None ):
        '''
        batch - Integer, index of batch within the current epoch.
        '''
        self.n_done += 1 
        
        if self.n_done % self.frequency != 0:
            return 
        
        if not self.on_train_batch_end_fn is None:
            self.on_train_batch_end_fn( batch=batch , logs=logs )
        

class ModelCheckpoint( Callback ):
    def __init__(self , checkpoint_path,  frequency=1 , only_save_at_end=False , overwrite_epochs=False ):
        super(ModelCheckpoint, self).__init__()
        self.frequency = frequency 
        self.checkpoint_path = checkpoint_path 
        self.only_save_at_end = only_save_at_end 
        self.overwrite_epochs = overwrite_epochs 
        
    def on_epoch_end( self , epoch , logs=None):
        
        if self.only_save_at_end :
            return 
        
        
        
        if (epoch+1)%self.frequency == 0:
            if self.overwrite_epochs :
                model_path = self.checkpoint_path + "_weights." + str( 0 )
            else:
                model_path = self.checkpoint_path + "_weights." + str( epoch )
            print("saving model " + model_path)
            self.model.save_weights(model_path  )
            
    def on_train_end( self , logs=None ):
        model_path = self.checkpoint_path + "_weights.final" 
        print("saving model final " + model_path)
        self.model.save_weights(model_path  )
        
    
        
class TensorboardCallback( Callback ):
    def __init__(self):
        super(TensorboardCallback, self).__init__()
        # todo 



