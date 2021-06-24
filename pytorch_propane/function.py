
from six import string_types
import inspect 
import sys
import torch 
import glob 

from .registry import registry
from .utils import filter_functions_kwargs , get_cli_opts , get_function_args 

import os 
import yaml 


def get_latest_epoch_no( checkpoint_path):
    all_weigths =  glob.glob(checkpoint_path+"_weights.*"  )
    all_epochs = [ p.replace(checkpoint_path+"_weights." , "") for p in all_weigths ]
    all_epochs = [ int( p ) for p in all_epochs if p != "final"]
    return all_epochs 

def load_checkpoints_weights( model , checkpoint_path , load_checkpoints_epoch=-1 , load_latest=False   ):
    # if checkpoins epochs is not -1 then select the final epoch 

    if load_checkpoints_epoch >= 0:
        model.load_weights( checkpoint_path+"_weights." + str( load_checkpoints_epoch ) ) 
        print("loaded weights " , checkpoint_path+"_weights." + str( load_checkpoints_epoch )  )
    else:
        if os.path.exists( checkpoint_path+"_weights.final"  ):
            model.load_weights( checkpoint_path+"_weights.final" )
            print("loaded weights " , checkpoint_path+"_weights.final"   )
        elif load_latest:
            load_checkpoints_epoch = get_latest_epoch_no(checkpoint_path  )
            model.load_weights( checkpoint_path+"_weights." + str( load_checkpoints_epoch ) ) 
            print("loaded weights " , checkpoint_path+"_weights." + str( load_checkpoints_epoch )  )
            
        else:
            raise ValueError("please provide an epoch number to load or set the load_latest true. ")
    
    




def get_model_from_checkpoint( load_checkpoint_path , return_function=False  ,  load_checkpoints_epoch=-1 , load_latest=False  ):

    model_config_path = load_checkpoint_path + "_model_config.yaml"
    model_config = yaml.safe_load(open(model_config_path))


    model_name = model_config['model_name']
    del model_config['model_name']

    if "network_name" in model_config:
        network_name = model_config['network_name']
        del  model_config['network_name'] 
    else:
        network_name = None 

    if 'network_config' in model_config:
        network_config = model_config['network_config']
        del model_config['network_config']

        if network_config is None:
            network_config = {}

    else:
        network_config = {}

    if not network_name is None:
        network , __network_kwargs  = get_network_object( network_name , **network_config )
    else:
        network = None 

    

    model_fn  = registry.get_model( model_name )

    if network is None:
        model = model_fn(**model_config)
    else:
        model = model_fn(network=network , **model_config)

    load_checkpoints_weights( model , load_checkpoint_path , load_checkpoints_epoch=load_checkpoints_epoch , load_latest=load_latest    )

    if return_function:
        return model , model_fn 
    else:
        return model 


    
def get_dataloader_from_checkpoint(load_checkpoint_path , eval_dataloader=False ):
    config_path = load_checkpoint_path + "_config.yaml"
    config = yaml.safe_load(open(config_path))
    config_objects = Function().__call__(just_return_objects=True , ** config )
    
    if eval_dataloader:
        return config_objects['eval_dataloader']
    else:
        return config_objects['dataloader']






def get_model_object(model_name=None , load_checkpoint_path=None , load_checkpoints_epoch=None , network=None  , **kwargs ):
    """Takes the model_name or the checkpoints_path , or both and return a model object and the filtered model_kwargs from kwargs 

    Args:
        kwargs ([type]): [description]
        model_name ([type], optional): [description]. Defaults to None.
        load_checkpoint_path ([type], optional): [description]. Defaults to None.
        load_checkpoints_epoch ([type], optional): [description]. Defaults to None.

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
        if network is None:
            model = model_fn(**model_kwargs)
        else:
            model = model_fn(**model_kwargs , network=network ) 
    

    if not load_checkpoint_path is None:
        if model is None:
            model , model_fn = get_model_from_checkpoint( load_checkpoint_path , return_function=True , load_checkpoints_epoch=load_checkpoints_epoch)
            model_kwargs = filter_functions_kwargs( model_fn , kwargs )
        else:
            load_checkpoints_weights( model , checkpoint_path=load_checkpoint_path , load_checkpoints_epoch=load_checkpoints_epoch )
            

    return model , model_kwargs 


def put_eval_args(dataloader_kwargs ,  dataloader_fn ,  kwargs ):
    eval_starting_kwargs = {} # the dictionary which needs to be put to the function , hence we remove the eval_ part 
    to_ret_eval_args = dataloader_kwargs.copy() # but we gotta return it with eval_ in it 
    
    for k in kwargs:
        if "eval_" in k:
            eval_starting_kwargs[ k.replace("eval_" , "") ] = kwargs[k ]
            to_ret_eval_args[k] = kwargs[k ]
    eval_starting_kwargs = filter_functions_kwargs( dataloader_fn , eval_starting_kwargs )
    for k in eval_starting_kwargs:
        dataloader_kwargs[k] = eval_starting_kwargs[ k ]
    return to_ret_eval_args 

def get_dataloader_object(dataloader_name=None , do_eval=False , **kwargs   ):
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
    # the above are the main dataloader args 
    if do_eval:
        # now we also find the eval dataloader args (starting with eval_ ) which should be overwritten 
        put_eval_args(dataloader_kwargs ,  dataloader_fn ,  kwargs )

    dataloader = dataloader_fn(**dataloader_kwargs)
    return dataloader , dataloader_kwargs 




def get_network_object(network_name=None , **kwargs  ):
    """Takes the network_name  return a network object and the filtered network_kwargs from kwargs 

    Args:
        network_name ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if network_name is None:
        raise ValueError("The function needs network and you have not provided any network info")
    
    network_fn = registry.get_network(network_name) 
    network_kwargs = filter_functions_kwargs( network_fn , kwargs )
    network = network_fn(**network_kwargs)
    return network , network_kwargs 




def get_dataset_object( dataset_name , do_eval=False , **kwargs ):
    if dataset_name is None:
        raise ValueError("The function needs dataloader and you have not provided any dataloader info")
    
    dataset_fn = registry.get_dataset(dataset_name) 
    dataset_kwargs = filter_functions_kwargs( dataset_fn , kwargs )
    # the above are the main dataloader args 
    if do_eval:
        # now we also find the eval dataloader args (starting with eval_ ) which should be overwritten 
        to_return_dataset_args = put_eval_args(dataset_kwargs ,  dataset_fn ,  kwargs )
    else:
        to_return_dataset_args = dataset_kwargs

    dataset = dataset_fn(**dataset_kwargs)
    return dataset , to_return_dataset_args 


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
    def __call__(self,  model=None  , model_name=None , load_checkpoint_path=None , load_checkpoints_epoch=None , 
        dataloader=None , dataloader_name=None , dataset=None , dataset_name=None , network=None ,  network_name=None , 
        eval_dataloader=None , eval_dataloader_name=None , eval_dataset=None  , eval_dataset_name=None  , 
        batch_size=None , eval_batch_size=None , data_num_workers=1 , eval_data_num_workers=1 , drop_last=False , just_return_objects=False  , **kwargs ) :
        """ This call function is used to transform the model_name , dataset_name , etc etc to model object , dataset object etc! 

        Args:
            model ([type], optional): [The model object ]. Defaults to None.
            model_name ([type], optional): [description]. Defaults to None.
            load_checkpoint_path ([type], optional): [description]. Defaults to None.
            load_checkpoints_epoch ([type], optional): [description]. Defaults to None.
            dataloader ([type], optional): [description]. Defaults to None.
            dataloader_name ([type], optional): [description]. Defaults to None.
            dataset ([type], optional): [description]. Defaults to None.
            dataset_name ([type], optional): [description]. Defaults to None.
            network ([type], optional): [description]. Defaults to None.
            network_name ([type], optional): [description]. Defaults to None.
            eval_dataloader ([type], optional): [description]. Defaults to None.
            eval_dataloader_name ([type], optional): [description]. Defaults to None.
            eval_dataset ([type], optional): [description]. Defaults to None.
            eval_dataset_name ([type], optional): [description]. Defaults to None.
            batch_size ([type], optional): [description]. Defaults to None.
            eval_batch_size ([type], optional): [description]. Defaults to None.
            data_num_workers (int, optional): [description]. Defaults to 1.
            eval_data_num_workers (int, optional): [description]. Defaults to 1.
            drop_last (bool, optional): [description]. Defaults to False.
            just_return_objects (bool, optional): [If this is set to true then it wont run self.exicute but just return the model, dataloader objects etc etc ]. Defaults to False.

        Returns:
            [type]: [description]
        """

        localss = locals()
        function_args = { arg: localss[arg] for arg in inspect.getfullargspec(self.__call__ ).args if arg != 'self'}  

        # inspect the functions and see if the model , dataloader etc are mandatory for the fucntion of not 
        function_needs_model = ( 'model' in get_function_args(self.execute)[2]   )
        function_needs_dataloader =  ( 'dataloader' in get_function_args(self.execute)[2]   )
        function_needs_eval_dataloader = ( 'eval_dataloader' in get_function_args(self.execute)[2]   )

        model_kwargs = {}
        dataloader_kwargs={}
        network_kwargs={}
        eval_dataloader_kwargs={}
        dataset_kwargs = {}
        eval_dataset_kwargs = {}


        if network is None and (not network_name is None ) :
            network , network_kwargs  = get_network_object( network_name , **kwargs )


        assert (not isinstance(model  , string_types)  ),  "ensure that model shold actually be a model object"

        if model is None  and ( (not model_name is None ) or ( not load_checkpoint_path is None  ) ) :
            model , model_kwargs = get_model_object(model_name=model_name , load_checkpoint_path=load_checkpoint_path , load_checkpoints_epoch=load_checkpoints_epoch , network=network  , **kwargs )


        # if the dataset is provided then  it will create the dataloader itself, hence the dataloader should not be provided 
        if ( not dataset_name is None ) or ( not dataset is None ):
            assert dataloader is None and dataloader_name is None , "If you provide a dataset then dont provide a dataloader "

        if ( not eval_dataset_name is None ) or ( not eval_dataset is None ):
            assert eval_dataloader is None and eval_dataloader_name is None , "If you provide a dataset then dont provide a dataloader "

        
        if not dataset_name is None:
            dataset , dataset_kwargs = get_dataset_object( dataset_name , **kwargs )
            assert not batch_size is None , "Please provide the batch size "
            dataloader = torch.utils.data.DataLoader(  dataset ,  batch_size=batch_size ,  shuffle=True, num_workers=data_num_workers, drop_last=drop_last  )

        if not eval_dataset_name is None:
            eval_dataset , eval_dataset_kwargs = get_dataset_object( eval_dataset_name, do_eval=True  , **kwargs  )
            assert not eval_batch_size is None , "Please provide the batch size "
            eval_dataloader = torch.utils.data.DataLoader(  eval_dataset ,  batch_size=eval_batch_size ,  shuffle=False , num_workers=eval_data_num_workers, drop_last=drop_last  )


        if dataloader is None and (not dataloader_name is None ) :
            dataloader , dataloader_kwargs = get_dataloader_object(dataloader_name=dataloader_name , **kwargs  )

        if eval_dataloader is None and (not eval_dataloader_name is None ) :
            eval_dataloader , eval_dataloader_kwargs = get_dataloader_object(dataloader_name=eval_dataloader_name, do_eval=True  , **kwargs  )

        # we dont want the kwargs which are sent to model/datalader to be sent to the function 
        non_function_keys = set( list(network_kwargs.keys()) +  list(eval_dataloader_kwargs.keys()) + list(dataloader_kwargs.keys() )
            + list(model_kwargs.keys()) + list(dataset_kwargs.keys()) + list(eval_dataset_kwargs.keys()) ) 
        fn_kwargs = {}
        for k in kwargs:
            if not k in non_function_keys:
                fn_kwargs[k] = kwargs[k ]
        
        # we have transformed model/dataloder names etc to model/datalodaer objects. now lets add them to a dict to pass to the execute functions 
        object_args = {} 
        if function_needs_eval_dataloader:
            assert not eval_dataloader is None , "eval dataloader should not be none "
        if not eval_dataloader is None:
            object_args["eval_dataloader"] = eval_dataloader

        if function_needs_dataloader:
            assert not dataloader is None , "dataloader should not be none "
        if not dataloader is None:
            object_args["dataloader"] = dataloader

        if function_needs_model:
            assert not model is None, "model should not be none "
        if not model is None:
            object_args["model"] = model 

        
        self.function_args_ser = kwargs.copy() # serliazable function arguments 
        self.function_args_ser.update( function_args ) # also add the function argumets 
        # now remove all the non serializable args 
        non_ser_args = ['model',  'dataloader',  'network' ,  'eval_dataloader' ,  'eval_dataset' ,  'just_return_objects' ]
        for arg in non_ser_args:
            del self.function_args_ser[arg]


        self.model_args = model_kwargs.copy()
        self.model_args['model_name'] = model_name 
        self.model_args['network_name'] = network_name 
        self.model_args['network_config'] = network_kwargs

        if just_return_objects:
            return object_args 
        else:
            out =  self.execute( **object_args  , **fn_kwargs )
            return out 

    # this one will be used by the cli engine 
    def _call_cli(self):
        pass
        # this should first detect all the args in the cli and then pass it to the call method! 

        args_dict = get_cli_opts( sys.argv )
        print("cli args " , args_dict )
        self.__call__(**args_dict )


        # or another option is to first inspect all the args from the models functions etc and then dynamically add them to the argmarse module 

        
