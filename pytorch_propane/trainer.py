# this module provides a high level training method, which saves model etc etc etc . This is different from the model.fit 

from six import string_types
import inspect 


def wrap_model_data_accept_function( fn ):
    """This is used as a decorator so that functions can accept model is an argument and use can provide things like checkpoints etc 

    Args:
        fn (function): [description]
    """

    # inspect the functions and see if the model needs some argument or not 
    function_needs_model = ( 'model' in inspect.getargspec( fn  ).args )
    function_needs_dataloader = ( 'dataloader' in inspect.getargspec( fn  ).args )
    function_needs_eval_dataloader = ( 'eval_dataloader' in inspect.getargspec( fn  ).args )

    def new_train_fn(  model=None  , model_name=None , load_checkpoint_path=None , checkpoints_epoch=None , 
        dataloader=None , dataloader_name=None , 
        eval_dataloader=None , eval_dataloader_name=None  , **kwargs  ):

        assert (not isinstance(model  , string_types)  ) # ensure that model shold actually be a model object haha 

        model_kwargs = {} # at first assume user wont provide the any model kwargs inside kwargs , also model kwargs should only be provided if model name is not none
        dataloader_kwargs = {}
        eval_dataloader_kwargs = {}

        if model is None  and function_needs_model :

            # load the model if the model name is provided
            if not model_name is None:
                model_fn  = registry.get_model( model_name )
                model_kwargs = filter_functions_kwargs( model_fn , kwargs )
                model = model_fn(**model_kwargs)
            

            if not load_checkpoint_path is None:
                if model is None:
                    model = get_model_from_checkpoint( load_checkpoint_path )
                else:
                    load_checkpoints_weights( model , checkpoint_path=load_checkpoint_path , checkpoints_epoch=checkpoints_epoch )

            else:
                if model_name is None:
                    raise ValueError("model_name cant be none if model or checkpoint path is not provided")
        
        if dataloader is None and function_needs_dataloader :
            
            if dataloader_name is None:
                raise ValueError("The function needs dataloader and you have not provided any dataloader info")
            
            dataloader_fn = registry.get_dataloader(dataloader_name) 
            dataloader_kwargs = filter_functions_kwargs( dataloader_fn , kwargs )
            dataloader = dataloader_fn(**dataloader_kwargs)

        if eval_dataloader is None and function_needs_eval_dataloader :
            
            if eval_dataloader_name is None:
                raise ValueError("The function needs dataloader and you have not provided any dataloader info")
            
            eval_dataloader_fn = registry.get_dataloader(eval_dataloader_name) 
            eval_dataloader_kwargs = filter_functions_kwargs( eval_dataloader_fn , kwargs , mandatory_prefix="eval_" )
            eval_dataloader = eval_dataloader_fn(**eval_dataloader_kwargs)

        # we dont want the kwargs which are sent to model , datalader to be sent to the function 
        non_function_keys = set( eval_dataloader_kwargs.keys() + dataloader_kwargs.keys() + model_kwargs.keys() ) 
        fn_kwargs = {}
        for k in kwargs:
            if not k in non_function_keys:
                fn_kwargs[k] = kwargs[k ]
        
        d = {}
        if function_needs_eval_dataloader:
            assert not eval_dataloader is None
            d[eval_dataloader] = eval_dataloader

        if function_needs_dataloader:
            assert not dataloader is None
            d[dataloader] = dataloader

        if function_needs_model:
            assert not model is None
            d[model] = model 
        
        fn( **d  , **fn_kwargs )

@wrap_model_data_accept_function 
def train(
    model 
):

class Trainer:
    def __init__(self):
        pass

    