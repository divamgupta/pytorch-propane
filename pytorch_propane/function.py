
from six import string_types
import inspect 
import sys
import torch 
import glob 

from .registry import registry
from .utils import filter_functions_kwargs , get_cli_opts , get_function_args 

import os 


def get_latest_epoch_no( checkpoint_path):
    all_weigths =  glob.glob(checkpoint_path+"_weights.*"  )
    all_epochs = [ p.replace(checkpoint_path+"_weights." , "") for p in all_weigths ]
    all_epochs = [ int( p ) for p in all_epochs if p != "final"]
    return all_epochs 

def load_checkpoints_weights( model , checkpoint_path , checkpoints_epoch=-1 , load_latest=False   ):
    # if checkpoins epochs is not -1 then select the final epoch 

    if checkpoints_epoch >= 0:
        model.load_weights( checkpoint_path+"_weights." + str( checkpoints_epoch ) ) 
        print("loaded weights " , checkpoint_path+"_weights." + str( checkpoints_epoch )  )
    else:
        if os.path.exists( checkpoint_path+"_weights.final"  ):
            model.load_weights( checkpoint_path+"_weights.final" )
            print("loaded weights " , checkpoint_path+"_weights.final"   )
        elif load_latest:
            checkpoints_epoch = get_latest_epoch_no(checkpoint_path  )
            model.load_weights( checkpoint_path+"_weights." + str( checkpoints_epoch ) ) 
            print("loaded weights " , checkpoint_path+"_weights." + str( checkpoints_epoch )  )
            
        else:
            raise ValueError("please provide an epoch number to load or set the load_latest true. ")
    
    




def get_model_from_checkpoint( load_checkpoint_path , return_function=True ,  checkpoints_epoch=-1 , load_latest=False  ):
    pass 




def get_model_object(model_name=None , load_checkpoint_path=None , checkpoints_epoch=None  , **kwargs ):
    """Takes the model_name or the checkpoints_path , or both and return a model object and the filtered model_kwargs from kwargs 

    Args:
        kwargs ([type]): [description]
        model_name ([type], optional): [description]. Defaults to None.
        load_checkpoint_path ([type], optional): [description]. Defaults to None.
        checkpoints_epoch ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """


    if model_name is None and load_checkpoint_path is None:
        raise ValueError("model_name cant be none if model or checkpoint path is not provided")
    
    # load the model if the model name is provided
    if not model_name is None:
        model_fn  = registry.get_model( model_name )
        model_kwargs = filter_functions_kwargs( model_fn , kwargs )
        model = model_fn(**model_kwargs)
    

    if not load_checkpoint_path is None:
        if model is None:
            model , model_fn = get_model_from_checkpoint( load_checkpoint_path , return_function=True , checkpoints_epoch=checkpoints_epoch)
            model_kwargs = filter_functions_kwargs( model_fn , kwargs )
        else:
            load_checkpoints_weights( model , checkpoint_path=load_checkpoint_path , checkpoints_epoch=checkpoints_epoch )
            

    return model , model_kwargs 

def get_dataloader_object(dataloader_name=None , **kwargs  ):
    """Takes the dataloader_name  return a dataloader object and the filtered dataloader_kwargs from kwargs 

    Args:
        dataloader_name ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if dataloader_name is None:
        raise ValueError("The function needs dataloader and you have not provided any dataloader info")
    
    dataloader_fn = registry.get_dataloader(dataloader_name) 
    dataloader_kwargs = filter_functions_kwargs( dataloader_fn , kwargs )
    dataloader = dataloader_fn(**dataloader_kwargs)
    return dataloader , dataloader_kwargs 

def get_dataset_object( dataset_name , **kwargs ):
    if dataset_name is None:
        raise ValueError("The function needs dataloader and you have not provided any dataloader info")
    
    dataset_fn = registry.get_dataset(dataset_name) 
    dataset_kwargs = filter_functions_kwargs( dataset_fn , kwargs )
    dataset = dataset_fn(**dataset_kwargs)
    return dataset , dataset_kwargs 


# this is the Function base class where users can inherit the Functions class 
# after inherit the user can fill the execute function where they can take model, dataloader objects as input 
# and now the __call__ is a wrapper which takes things like model_name string etc and converts to the twmele Model object so that the user does not have to do that manually 
# and this class also creates a cli version of the function as well! 
# if the user does not give the dataloader but gives the dataset then propane will automatically create the dataloader from given dataset or the dataset class 
# todo give support for the network also , so that user can just specify the pytorch  network name not the propane model! 
class Function:
    def __init__(self):
        pass
    
    # you can override this function and put whatever you wanna put 
    def execute():
        raise NotImplementedError("This needs to be overridden")

    # do not override this one tho ! 
    def __call__(self,  model=None  , model_name=None , load_checkpoint_path=None , checkpoints_epoch=None , 
        dataloader=None , dataloader_name=None , dataset=None , dataset_name=None , 
        eval_dataloader=None , eval_dataloader_name=None , eval_dataset=None  , eval_dataset_name=None  , 
        batch_size=None , eval_batch_size=None , data_num_workers=1 , eval_data_num_workers=1 , drop_last=False   , **kwargs ) :

        # inspect the functions and see if the model , dataloader etc are mandatory for the fucntion of not 
        function_needs_model = ( 'model' in get_function_args(self.execute)[2]   )
        function_needs_dataloader =  ( 'dataloader' in get_function_args(self.execute)[2]   )
        function_needs_eval_dataloader = ( 'eval_dataloader' in get_function_args(self.execute)[2]   )

        model_kwargs = {}
        dataloader_kwargs={}
        eval_dataloader_kwargs={}
        dataset_kwargs = {}
        eval_dataset_kwargs = {}



        assert (not isinstance(model  , string_types)  ) # ensure that model shold actually be a model object haha 

        if model is None  and ( (not model_name is None ) or ( not load_checkpoint_path is None  ) ) :
            model , model_kwargs = get_model_object(model_name=model_name , load_checkpoint_path=load_checkpoint_path , checkpoints_epoch=checkpoints_epoch  , **kwargs )



        # if the dataset is provided then  it will create the dataloader itself, hence the dataloader should not be provided 
        if ( not dataset_name is None ) or ( not dataset is None ):
            assert dataloader is None and dataloader_name is None , "If you provide a dataset then dont provide a dataloader "

        if ( not eval_dataset_name is None ) or ( not eval_dataset is None ):
            assert eval_dataloader is None and eval_dataloader_name is None , "If you provide a dataset then dont provide a dataloader "

        
        if not dataset_name is None:
            dataset , dataset_kwargs = get_dataset_object( dataset_name , **kwargs )
            assert not batch_size is None 
            dataloader = torch.utils.data.DataLoader(  dataset ,  batch_size=batch_size ,  shuffle= True, num_workers=data_num_workers, drop_last=drop_last  )

        if not eval_dataset_name is None:
            eval_dataset , eval_dataset_kwargs = get_dataset_object( eval_dataset_name , **kwargs )
            assert not eval_batch_size is None 
            eval_dataloader = torch.utils.data.DataLoader(  eval_dataset ,  batch_size=eval_batch_size ,  shuffle=False , num_workers=eval_data_num_workers, drop_last=drop_last  )


        if dataloader is None and (not dataloader_name is None ) :
            dataloader , dataloader_kwargs = get_dataloader_object(dataloader_name=dataloader_name , **kwargs  )

        if eval_dataloader is None and (not eval_dataloader_name is None ) :
            eval_dataloader , eval_dataloader_kwargs = get_dataloader_object(dataloader_name=eval_dataloader_name , **kwargs  )

        # we dont want the kwargs which are sent to model/datalader to be sent to the function 
        non_function_keys = set( list(eval_dataloader_kwargs.keys()) + list(dataloader_kwargs.keys() )+ list(model_kwargs.keys()) + list(dataset_kwargs.keys()) + list(eval_dataset_kwargs.keys()) ) 
        fn_kwargs = {}
        for k in kwargs:
            if not k in non_function_keys:
                fn_kwargs[k] = kwargs[k ]
        
        # we have transformed model/dataloder names etc to model/datalodaer objects. now lets add them to a dict to pass to the execute functions 
        object_args = {} 
        if function_needs_eval_dataloader:
            assert not eval_dataloader is None
        if not eval_dataloader is None:
            object_args["eval_dataloader"] = eval_dataloader

        if function_needs_dataloader:
            assert not dataloader is None
        if not dataloader is None:
            object_args["dataloader"] = dataloader

        if function_needs_model:
            assert not model is None
        if not model is None:
            object_args["model"] = model 

        
        self.function_args_ser = kwargs.copy() # serliazable function arguments 
        self.function_args_ser['model_name'] = model_name 
        self.model_args = model_kwargs 

        out =  self.execute( **object_args  , **fn_kwargs )

        return out 

    # this one will be used by the cli engine 
    def _call_cli(self):
        pass
        # this should first detect all the args in the cli and then pass it to the call method! 

        args_dict = get_cli_opts( sys.argv )
        self.__call__(**args_dict )


        # or another option is to first inspect all the args from the models functions etc and then dynamically add them to the argmarse module 

        
