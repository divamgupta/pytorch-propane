# this module provides a high level training method, which saves model etc etc etc . This is different from the model.fit 

from six import string_types
import inspect 



def filter_functions_kwargs():
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
            model = get_model_from_checkpoint( load_checkpoint_path )
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


# this is the Function base class where users can inherit the Functions class 
# after inherit the user can fill the execute function where they can take model, dataloader objects as input 
# and now the __call__ is a wrapper which takes things like model_name string etc and converts to the model object so that the user does not have to do that manually 
# and this class also creates a cli version of the function as well! 
class Function:
    def __init__(self):
        pass
    
    # you can override this function and put whatever you wanna put 
    def execute():
        raise NotImplementedError("This needs to be overridden")

    # do not override this one tho ! 
    def __call__(self,  model=None  , model_name=None , load_checkpoint_path=None , checkpoints_epoch=None , 
        dataloader=None , dataloader_name=None , 
        eval_dataloader=None , eval_dataloader_name=None  , **kwargs ) :
        # inspect the functions and see if the model needs some argument or not 
        function_needs_model = ( 'model' in inspect.getargspec( self.execute  ).args )
        function_needs_dataloader = ( 'dataloader' in inspect.getargspec( self.execute  ).args )
        function_needs_eval_dataloader = ( 'eval_dataloader' in inspect.getargspec( self.execute  ).args )

        model_kwargs = {}
        dataloader_kwargs={}
        eval_dataloader_kwargs={}

        assert (not isinstance(model  , string_types)  ) # ensure that model shold actually be a model object haha 

        if model is None  and function_needs_model :
            model , model_kwargs = get_model_object(model_name=model_name , load_checkpoint_path=load_checkpoint_path , checkpoints_epoch=checkpoints_epoch  , **kwargs )

        if dataloader is None and function_needs_dataloader :
            dataloader , dataloader_kwargs = get_dataloader_object(dataloader_name=dataloader_name , **kwargs  )

        if eval_dataloader is None and function_needs_eval_dataloader :
            eval_dataloader , eval_dataloader_kwargs = get_dataloader_object(dataloader_name=eval_dataloader_name , **kwargs  )

        # we dont want the kwargs which are sent to model/datalader to be sent to the function 
        non_function_keys = set( eval_dataloader_kwargs.keys() + dataloader_kwargs.keys() + model_kwargs.keys() ) 
        fn_kwargs = {}
        for k in kwargs:
            if not k in non_function_keys:
                fn_kwargs[k] = kwargs[k ]
        
        # we have transformed model/dataloder names etc to model/datalodaer objects. now lets add them to a dict to pass to the execute functions 
        object_args = {} 
        if function_needs_eval_dataloader:
            assert not eval_dataloader is None
            object_args[eval_dataloader] = eval_dataloader

        if function_needs_dataloade:
            assert not dataloader is None
            object_args[dataloader] = dataloader

        if function_needs_model:
            assert not model is None
            object_args[model] = model 
        
        return self.execute( **object_args  , **fn_kwargs )

    # this one will be used by the cli engine 
    def _call_cli(self):
        pass


# todo have a function decorator which returns this function object actually  , to transform the def function to the an instance of a class object with the __call__ method

def register_functions(name=None):

    # just transform it but dont add it to the database
    if name is None:
        pass

@register_function 
class Trainer(Function):
    def __init__(self):
        pass


@register_function(name="train" )
def train(
    model 
):
    pass 

# this register function is all the user neeeds to put, whether over a class Function or a def function .. it will figure out if its a fn or class , and register it 
# this can be declared in the registry only 
# you should be able to register these globally or you can register these only to a model.. say you wanna have diffrent trains for diffrnet models 