
from .function import Function
from .utils import  get_cli_opts 
import sys 
from .registry import registry
from . import trainer 
import os 

def run_cli():


    if len( sys.argv ) < 3:
        raise Exception("The format is pytorch_propane <module_name> <function_name> <function args> ")

    module_name = sys.argv[1]
    function_name = sys.argv[2 ]

    #  the user provides the module name wehre all the models etc are defined 
    # also the user would neeed to register all the models , dataloaders etc in that module 
    sys.path.append( os.getcwd() )

    __import__( module_name , globals(), locals()) 

    function = registry.get_function( function_name )

    args_dict = get_cli_opts( sys.argv )
    print("cli args " , args_dict )

    function()._call_cli( args_dict=args_dict  )
    
    
     