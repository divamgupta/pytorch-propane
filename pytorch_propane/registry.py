# contains code to register the models, dataloaders etc 
from functools import wraps
from six import string_types



class Registry():
    def __init__(self):
        self.models = {}
        self.dataloaders = {}

    def register_model(self, model_fn , model_name ):
        self.models[model_name ] = model_fn 
    
    def register_dataloader(self , dataloader_fn , dataloader_name):
        self.dataloaders[dataloader_name] = dataloader_fn

    def get_model( self , model_name ):
        return self.models[model_name]

    def get_dataloader( self , dataloader_name):
        return self.dataloaders[ dataloader_name ]
    # the decorateors : 

    def model(self , model_name  ):
        def decorator(model_fn):

            @wraps( model_fn )
            def new_model_fn(*args , **kwargs):
                out = model_fn( *args , **kwargs )
                model_fn.model_name = model_name 
                model_fn.registry = self 
                return out 

            self.register_model(new_model_fn , model_name)
            new_model_fn.model_name = model_name 
            new_model_fn.registry = self 
            return new_model_fn 
        return decorator

    def add_model_function( self , function ):
        pass 

    def default_dataloader( self , dataloader ):
        if isinstance(dataloader , string_types):
            if not dataloader in self.dataloaders:
                raise ValueError("The dataloader named %d has not been registered."%dataloader )
            dataloader = self.dataloaders[dataloader]

        def decorator( model_fn ):
            model_fn.default_dataloader = dataloader
            return model_fn 

        return decorator 
    
    def dataloader( self , dataloader_name ):
        def decorator(dataloader_fn):
            self.register_dataloader(dataloader_fn , dataloader_name)
            dataloader_fn.model_name = dataloader_name 
            dataloader_fn.registry = self 
            return dataloader_fn  
        return decorator
    

    
