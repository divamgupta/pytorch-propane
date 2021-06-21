
import torch 
import numpy as np 
from six import string_types
from torch import optim
import inspect 
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm 
import copy 

def get_function_args( fn ):
    """returns a list of all argumnts, dict of all the defualts , and list of all non default arguments 

    Args:
        fn (function): [description]

    Returns:
        [type]: [description]
    """
    args = inspect.getargspec( fn  ).args 

    if inspect.getargspec( fn  ).defaults  is None:
        n_defaults = 0 
        def_args = []
    else:
        n_defaults = len(inspect.getargspec( fn  ).defaults  )
        def_args = list(inspect.getargspec( fn  ).defaults )

    if n_defaults > 0:
        default_args = args[ -1*n_defaults : ]
    else:
        default_args = []

    defaults = { a[0]:a[1] for a in zip(default_args , def_args  ) }
    non_defaults = args[: len( args) - n_defaults ]

    return args , defaults , non_defaults 


# given a dictionary kwargs .. this will return which all of those can be sent to the function fn_name  
def filter_functions_kwargs(fn_name , kwargs ):
    fn_args = inspect.getargspec( fn_name  ).args
    ret = {}
    for k in kwargs:
        if k in fn_args:
            ret[ k ] = kwargs[k]
    return ret  


def str_to_auto_type(var):
    #first test bools
    if var == 'True' or var=='true':
            return True
    elif var == 'False' or var=='false':
            return False
    else:
            #int
            try:
                    return int(var)
            except ValueError:
                    pass
            #float
            try:
                    return float(var)
            except ValueError:
                    pass

            # homogenus list 
            # todo 


            #string
            try:
                    return str(var)
            except ValueError:
                    raise NameError('Something Messed Up Autocasting var %s (%s)' 
                                      % (var, type(var)))


# returns a dictionarly of named args from cli!! 
def get_cli_opts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.

    argv= copy.deepcopy(argv)

    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-' and  argv[0][1] == '-':  # Found a "--name value" pair.

            argv[0] = argv[0][2:] # remove '--' 


            assert argv[0] != '' , "There is some issue with the cli args becasue a key cannot be empty"
            opts[argv[0]] = str_to_auto_type( argv[1] )   # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.

    return opts




        
    
def get_vars( data , cuda=False  , numpy=False ):
    
#     list( map( lambda x :Variable(torch.FloatTensor(x.float() )).cuda() , imgs   ))
    
    
    if type( data ) is tuple:
        return tuple([ get_vars(d , cuda=cuda , numpy=numpy) for d in data  ])
    elif type( data ) is list:
        return list([ get_vars(d , cuda=cuda , numpy=numpy) for d in data  ])
    elif type( data ) is dict:
        return { k:get_vars(data[k] , cuda=cuda , numpy=numpy) for k in data }
    else:
        if numpy:
            data = torch.from_numpy(data)
        r =  Variable( data )
        if cuda:
            r = r.cuda()
        return r
    
def get_np_arrs( data ):
    if type( data ) is tuple:
        return tuple([ get_np_arrs(d  ) for d in data  ])
    elif type( data ) is list:
        return list([ get_np_arrs(d  ) for d in data  ])
    elif type( data ) is dict:
        return { k:get_np_arrs(data[k]  ) for k in data }
    else:
        return data.cpu().detach().numpy()
    
    


class ProgressBar(tqdm):
    def __init__( self , iterator ):
        super(ProgressBar, self).__init__(iterator)
        self.vals_history_dict = {}
        
    def add( self , vals_dict ):
        for k in vals_dict:
            if not k in self.vals_history_dict:
                self.vals_history_dict[k] = []
            self.vals_history_dict[k].append( vals_dict[k])
        
        bar_str = ""
        for k in self.vals_history_dict:
            bar_str += k+":"+ "%.3f"%(np.mean(self.vals_history_dict[k])) + " "
        
        self.set_description(bar_str )
        
    


    