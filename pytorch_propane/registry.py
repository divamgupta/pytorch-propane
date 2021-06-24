# contains code to register the models, dataloaders etc 
from functools import wraps
from six import string_types
from decorator import decorator
import wrapt 
import inspect 
import types 



class Registry():
    def __init__(self):
        self.models = {}
        self.dataloaders = {}
        self.datasets = {}
        self.functions_dict = {}
        self.networks = {}
    # for registering the models 

    def _register_model(self, model_name , model_fn   ):
        self.models[model_name ] = model_fn 

    def _register_model_dec(self , model_name  ):

        def deccc(model_fn):
            print("called ")

            @wraps( model_fn  )
            def new_model_fn(*args , **kwargs):
                out = model_fn( *args , **kwargs )
                model_fn.model_name = model_name 
                model_fn.registry = self 
                return out 

            new_model_fn.__signature__ = inspect.signature(model_fn  ) 

            self._register_model(model_name=model_name , model_fn=new_model_fn )
            new_model_fn.model_name = model_name 
            new_model_fn.registry = self 
            return new_model_fn

        return deccc

    def register_model( self , model_name , model_fn=None):
        if model_fn is None:
            return self._register_model_dec( model_name )
        return self._register_model( model_name , model_fn ) 
    
    def get_model( self , model_name ):
        return self.models[model_name]
    

    def _register_dataloader(self , dataloader_fn , dataloader_name):
        self.dataloaders[dataloader_name] = dataloader_fn

    # a function decorator which uses should put above dataloader functions to regirter the dataloader given dataloader name 
    def _register_dataloader_dec( self , dataloader_name ):
        

        def decorator(dataloader_fn):
            self._register_dataloader(dataloader_fn=dataloader_fn , dataloader_name=dataloader_name )
            dataloader_fn.dataloader_name = dataloader_name 
            dataloader_fn.registry = self 

            return dataloader_fn  
        return decorator

    def register_dataloader(self ,  dataloader_name , dataloader_fn=None ):

        if dataloader_fn is None:
            return self._register_dataloader_dec( dataloader_name )
        return self._register_dataloader(dataloader_fn=dataloader_fn , dataloader_name=dataloader_name ) 


    def get_dataloader( self , dataloader_name):
        return self.dataloaders[ dataloader_name ]


    def _register_dataset(self , dataset_fn , dataset_name):
        self.datasets[dataset_name] = dataset_fn

        # a function decorator which uses should put above dataset functions to regirter the dataset given dataset name 
    def _register_dataset_dec( self , dataset_name ):

        def decorator(dataset_fn):
            self._register_dataset(dataset_fn=dataset_fn , dataset_name=dataset_name )
            dataset_fn.dataset_name = dataset_name 
            dataset_fn.registry = self 
            return dataset_fn  
        return decorator

    def register_dataset(self ,  dataset_name , dataset_fn=None ):

        if dataset_fn is None:
            return self._register_dataset_dec( dataset_name )
        return self._register_dataset(dataset_fn=dataset_fn , dataset_name=dataset_name )

    def get_dataset( self , dataset_name):
        return self.datasets[ dataset_name ]



    def _register_network(self , network_fn , network_name):
        self.networks[network_name] = network_fn

        # a function decorator which uses should put above network functions to regirter the network given network name 
    def _register_network_dec( self , network_name ):

        def decorator(network_fn):
            self._register_network(network_fn=network_fn , network_name=network_name )
            network_fn.network_name = network_name 
            network_fn.registry = self 
            return network_fn  
        return decorator

    def register_network(self ,  network_name , network_fn=None ):

        if network_fn is None:
            return self._register_network_dec( network_name )
        return self._register_network(network_fn=network_fn , network_name=network_name )

    def get_network( self , network_name):
        return self.networks[ network_name ]



    def _register_function(self , function , function_name):
        from pytorch_propane.function import Function

        if inspect.isclass( function ) and issubclass( function , Function ):
            self.functions_dict[function_name] = function
        elif isinstance( function , types.FunctionType):
            raise NotImplementedError("not implemented")
        else:
            raise NotImplementedError("not implemented")

        


        # a function decorator which uses should put above function functions to regirter the function given function name 
    def _register_function_dec( self , function_name ):

        def decorator(function):
            self._register_function(function=function , function_name=function_name )
            return function  
        return decorator

    def register_function(self ,  function_name , function=None ):

        if function is None:
            return self._register_function_dec( function_name )
        return self._register_function(function=function , function_name=function_name ) 

    def get_function(self , function_name ):
        return self.functions_dict[ function_name ]
        

    

    

    

    

    def add_model_function( self , function ):
        pass 

    # a decorator to attach a default dataloader to the model 
    def default_dataloader( self , dataloader ):
        if isinstance(dataloader , string_types):
            if not dataloader in self.dataloaders:
                raise ValueError("The dataloader named %d has not been registered."%dataloader )
            dataloader = self.dataloaders[dataloader]

        def decorator( model_fn ):
            model_fn.default_dataloader = dataloader
            return model_fn 

        return decorator 
    

    

    
registry = Registry() # the default registry which the user should use by defualt 