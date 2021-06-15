# this module provides a high level training method, which saves model etc etc etc . This is different from the model.fit 

from six import string_types
from .registry import registry
from .function import Function 
import yaml
import os 
import glob 

from .callbacks import ModelCheckpoint ,  Callback
from .function import get_model_from_checkpoint , load_checkpoints_weights 


class ModelTrainingStatusCallback( Callback ):
    """a callback which sets the status of the training in a file. the statuses are started meaning training in progress and finished meaning training ended 

    Args:
        Callback ([type]): [description]
    """
    def __init__(self , status_file_path ):
        super(ModelTrainingStatusCallback, self).__init__()
        self.status_file_path = status_file_path 

    def on_train_batch_end( self , batch , logs=None ):
        if batch > 1:
            open( self.status_file_path , 'w').write("started")


    def on_train_end( self , logs=None ):
        open( self.status_file_path , 'w').write("finished")

# status file -> starting , started , finished 

# @register_function 
class Trainer(Function):
    def __init__(self):
        pass

    def execute(self , model , dataloader , eval_dataloader=None , save_path=None , 
        load_path=None , load_epoch=-1 , overwrite_prev_training=False , n_epochs=1 , sanity=False , 
        save_frequency =1   , which_epochs_to_save=None , overwrite_epochs=False ,  auto_resume_checkpoint=False , 
         ):
        pass 


        # save the config etc 
        if not save_path is None:
            config_path = save_path + "_config.yaml"
            model_config_path = save_path + "__model_config.yaml"
            status_file_path = save_path + "_status.txt"

            # throw exception if training had already happened and finished for the given save path 
            if os.path.exists(status_file_path):
                prev_status = open( status_file_path).read()
                if prev_status == 'finished':
                    if not overwrite_prev_training:
                        raise Exception("Looks like training was finished at this checkpoint path " + save_path ) 

            
            open( status_file_path , 'w').write("starting")

            # save all the config files 
            yaml.dump(self.function_args_ser  , open(config_path), 'w')
            yaml.dump(self.model_args  , open(model_config_path), 'w')

            if not save_path is None:
                model.add_callback( ModelCheckpoint( save_path  , save_frequency  , overwrite_epochs=overwrite_epochs ) )

            model.add_callback( ModelTrainingStatusCallback(status_file_path) )

    
        if not load_path is None:
            load_checkpoints_weights( model , load_path , checkpoints_epoch=load_epoch  )

        # todo : add a callback which can add things like to log all the training in a file! 
    

        # start training 
        model.fit_dataset( dataloader , validation_data=eval_dataloader , epochs=n_epochs  , sanity=sanity  )
        

        # todo : do some eval and then save the results in another file 








# todo have a function decorator which returns this function object actually  , to transform the def function to the an instance of a class object with the __call__ method

def register_functions(name=None):

    # just transform it but dont add it to the database
    if name is None:
        pass




# @register_function(name="train" )
def train(
    model 
):
    pass 

# this register function is all the user neeeds to put, whether over a class Function or a def function .. it will figure out if its a fn or class , and register it 
# this can be declared in the registry only 
# you should be able to register these globally or you can register these only to a model.. say you wanna have diffrent trains for diffrnet models 


# IMP even if the model name etc has not been registerd to the registry ,,, the code shuld also map the model_name string via the functions definition 