# contains code to register the models, dataloaders etc 
from functools import wraps
from six import string_types



class Registry():
    def __init__(self):
        self.models = {}
        self.dataloaders = {}
        self.datasets = {}

    # for registering the models 

    def _register_model(self, model_name , model_fn   ):
        self.models[model_name ] = model_fn 

    def _register_model_dec(self , model_name  ):
        def decorator(model_fn):

            @wraps( model_fn )
            def new_model_fn(*args , **kwargs):
                out = model_fn( *args , **kwargs )
                model_fn.model_name = model_name 
                model_fn.registry = self 
                return out 

            self._register_model(model_name=model_name , new_model_fn=new_model_fn )
            new_model_fn.model_name = model_name 
            new_model_fn.registry = self 
            return new_model_fn 
        return decorator

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