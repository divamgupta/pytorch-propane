 
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


from .loss import Loss 
from .utils import get_vars , get_np_arrs  , ProgressBar
from .callbacks import Callback 

        
class Model:
    def __init__(self , network ):
        self.network = network 
        self.optimizer = None 
        self.loss_modules = []
        self.metric_modules_train = []
        self.metric_modules_eval = []
        self.callbacks = []
        self.cuda = False 
    
    
    def add_metric( self ,  metric , display_name=None  ,  output_key=None  , mode=None):
        '''
        
        add metrics to display which do not contribute to the loss gradients ! 
        
        output_key -> key / index of model output where the loss should be applied 
        display name -> the name of the loss which you have to display
        metric -> similar to the loss object 
        
        '''
        if not isinstance(metric , Loss ):
            metric = Loss(  loss_fn=metric , key=output_key  )
            
        if display_name is None:
            display_name = metric.display_name
        
        if mode is None or mode=='train':
            self.metric_modules_train.append( (metric , display_name ) )
        if mode is None or mode=='eval':
            self.metric_modules_eval.append( (metric , display_name ) )
            
    
    def add_loss( self , loss , output_key=None , display_name=None , loss_weight=1  ):
        """
        output_key -> key / index of model output where the loss should be applied 
        display name -> the name of the loss which you have to display
        loss -> the function / string of the loss 
        """
        
        if not isinstance(loss , Loss ):
            loss = Loss(  loss_fn=loss , key=output_key, loss_weight=loss_weight )
            
        if display_name is None:
            display_name = loss.display_name
            
        self.loss_modules.append( (loss , display_name ) )
        
    def set_optimiszer( self , optimizer ):
        opts_dict = { 'sgd':optim.SGD , 'adam' :optim.Adam , 'adadelta': optim.Adadelta , 'adagrad': optim.Adagrad , 'rmsprop':optim.RMSprop }
        
        if isinstance(optimizer , string_types):
            optimizer = opts_dict[ optimizer ]            
            
        if inspect.isclass(optimizer)  and  issubclass( optimizer , optim.Optimizer):
            optimizer = optimizer( self.network.parameters())
            
        self.optimizer = optimizer 
            
            
        
        
    def compute_loss( self ,  model_output=None ,  data_y=None  , data_x=None  ):
        '''
        aggrigate all the loss functions and apply it 
        '''
        loss_dict = {}
        total_loss = 0
        
        for loss_m , loss_disp_name  in self.loss_modules:
            loss_val = loss_m( model_output=model_output ,data_y=data_y , data_x=data_x  )
            total_loss += loss_val
            if not loss_disp_name is None:
                loss_dict[ loss_disp_name  ] = loss_val 
        return total_loss , loss_dict 
    
    def compute_metrics( self , model_output=None ,  data_y=None  , data_x=None  , mode='train' ):
        '''
        compute all the metrics LOL 
        '''
        loss_dict = {}
        total_loss = 0
        
        if mode == 'train':
            metric_modules = self.metric_modules_train
        elif mode == 'eval':
            metric_modules = self.metric_modules_eval
        
        for i , ( loss_m , loss_disp_name)  in enumerate( metric_modules ):
            if loss_disp_name is None:
                loss_disp_name = str( i )
            loss_dict[ loss_disp_name  ] = loss_m( model_output=model_output ,data_y=data_y , data_x=data_x  )

        return  loss_dict 
        
    
    
    
    def compile( self , optimizer=None  , loss=None , metrics=[] , loss_weights=None , cuda=False  , data_parallel=False  ):
        for metric in metrics:
            self.add_metric( metric )
            
        self.cuda = cuda 
        if data_parallel:
            self.network = nn.DataParallel(self.network)
        if cuda:
            self.network = self.network.cuda()
        
        if not loss is None:
            if (isinstance(loss, list)) or (isinstance(loss, tuple )):
                losses_itt = enumerate(loss)
            elif (isinstance(loss, dict)) :
                losses_itt = loss.items()
            else:
                losses_itt = None
            
            if not losses_itt is None:
                for i,l in losses_itt :
                    if loss_weights is None:
                        l_w = 1
                    else:
                        l_w = loss_weights[i]
                    self.add_loss( l , output_key=i , loss_weight=l_w )
            else:
                self.add_loss( loss )
        
        if not optimizer is None:
            self.set_optimiszer( optimizer )
                        
                    
    
    
    def train_step(self, data_x=None , data_y=None ):
        if len( self.loss_modules ) == 0:
            raise ValueError("No losses attached.")
        if self.optimizer is None:
            raise ValueError("No optimizer attached.")
            
        self.network.train()

        if not data_x is None:
            data_x = get_vars( data_x , cuda=self.cuda , numpy=False  )

        if not data_y is None:
            data_y = get_vars( data_y , cuda=self.cuda , numpy=False  )

        self.optimizer.zero_grad()
        model_output = self.network( data_x )

        total_loss , loss_dict  = self.compute_loss( model_output=model_output ,  data_y=data_y  , data_x=data_x )
        metrics_dict = self.compute_metrics( model_output=model_output ,  data_y=data_y  , data_x=data_x , mode='train')
        
        
        total_loss.backward()
        self.optimizer.step()

        return total_loss , loss_dict , metrics_dict  


    def predict_on_batch( self, data_x , numpy=False  ):
        data_x = get_vars( data_x  , cuda=self.cuda , numpy=numpy )
        model_output = self.network( data_x )
        if numpy:
            model_output = get_np_arrs(model_output)
        return model_output

        
    
    def valid_step(self , data_x=None , data_y=None  ):
        self.network.eval()
        
        if not data_x is None:
            data_x = get_vars( data_x  , cuda=self.cuda , numpy=False )

        if not data_y is None:
            data_y = get_vars( data_y , cuda=self.cuda , numpy=False  )
        
        model_output = self.network( data_x )
        total_loss , loss_dict  = self.compute_loss( model_output=model_output ,  data_y=data_y  , data_x=data_x )
        metrics_dict = self.compute_metrics( model_output=model_output ,  data_y=data_y  , data_x=data_x , mode='eval')
        
        return total_loss , loss_dict , metrics_dict  
        
    
    def add_callback( self , callback  ):
        self.callbacks.append( callback )
        
    def add_callback_fn( self , callback_fn , event , frequency=1 ):
        self.add_callback( Callback( callback_fn=callback_fn , event=event , frequency=frequency ) )
        
    def call_callbacks(self , event, batch_id=None , logs=None ):
        
        for callback in self.callbacks:
                
            callback.model = self 
            
            if event == "on_train_batch_end":
                callback.on_train_batch_end( batch=batch_id,  logs=logs)  
            elif event == "on_epoch_end":
                callback.on_epoch_end( epoch=batch_id ,  logs=logs) 
            elif event == "on_train_end":
                callback.on_train_end( logs=logs)   
            else:
                raise ValueError("invalid event %s"%event ) 
                
                
    def evaluate_dataset( self , validation_data , verbose=1 , sanity=False ):
        if verbose == 1:
            print("validation:")
            pbar = ProgressBar(validation_data )
        else:
            pbar = validation_data 

        for batch_idx, ( data_x , data_y ) in enumerate(pbar):
            total_loss , loss_dict , metrics_dict = self.valid_step(data_x=data_x , data_y=data_y)
#             print('Val '  , 'Iter' ,batch_idx, "loss" , total_loss.item()  )
            if verbose == 1:
                loss_dict_a = {k:loss_dict[k].item() for k in loss_dict}
                loss_dict_a.update({k:metrics_dict[k].item() for k in metrics_dict})
                loss_dict_a['val_loss_total'] = total_loss.item()
                pbar.add(loss_dict_a)
                  
            if sanity and batch_idx>=3:
                break 
                
    def run_inference_callback( self , data , callback , output_key=None , verbose=1  ,batch_wise=False ,  sanity=False ):
        self.network.eval()
        assert not batch_wise , "not implemtted "
        callback.on_start()
        
        if verbose == 1:
            pbar = tqdm( data )
        else:
            pbar = data 
        
        for batch_idx, ( data_x , data_y ) in enumerate(pbar):
            
            if not data_x is None:
                data_x = get_vars( data_x , cuda=self.cuda , numpy=False  )

            if not data_y is None:
                data_y = get_vars( data_y , cuda=self.cuda , numpy=False  )


            model_output = self.network( data_x )
            if output_key is None:
                callback( data_x=data_x , data_y=data_y , model_output=model_output )
            else:
                callback( data_x=data_x , data_y=data_y[output_key] , model_output=model_output[output_key] )
            if sanity and batch_idx>=3:
                break                 
        callback.on_end()     
            
        
    
    def fit_dataset(self , training_data,  epochs=1, verbose=1, callbacks=None, validation_data=None , validation_freq=1 , sanity=False  ):
        
        if sanity:
            epochs = min( sanity , 3 )
        
        total_iter_done = 0 
        for epoch in range(epochs) :
            if verbose == 1:
                print("epoch %d : "%epoch)
                pbar = ProgressBar(training_data )
            else:
                pbar = training_data 
                
            for batch_idx, ( data_x , data_y ) in enumerate(pbar):
                total_iter_done += 1 
                total_loss , loss_dict , metrics_dict = self.train_step(data_x=data_x , data_y=data_y)
                #print('Train Epoch' , epoch , 'Iter' ,batch_idx, "loss" , total_loss.item()  )
                
                if verbose == 1:
                    loss_dict_a = {k:loss_dict[k].item() for k in loss_dict}
                    loss_dict_a['train_loss_total'] = total_loss.item()
                    pbar.add(loss_dict_a)
                
                self.call_callbacks('on_train_batch_end' , batch_id=total_iter_done )
                
                if sanity and batch_idx>=3:
                    break 
                    
            if ( not validation_data is None ) and   epoch%validation_freq == 0:
                self.evaluate_dataset( validation_data , verbose=verbose , sanity=sanity )
                
            self.call_callbacks('on_epoch_end' , batch_id=epoch )
        self.call_callbacks('on_train_end')
        
            
    
    def save_weights(self , weights_path ):
        torch.save(self.network.state_dict(), weights_path) 
    
    def load_weights( self , weights_path ):
        self.network.load_state_dict(torch.load(weights_path))
    
    
    
