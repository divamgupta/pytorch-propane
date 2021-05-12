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

    # the decorateors : 

    def model(self , model_name  ):
        def decorator(model_fn):
            self.register_model(model_fn , model_name)
            model_fn.model_name = model_name 
            model_fn.registry = self 
            return model_fn 
        return decorator

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
    

    
