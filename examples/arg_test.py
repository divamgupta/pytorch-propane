# import argparse
# parser = argparse.ArgumentParser()
# args, unknown = parser.parse_known_args()
# print( args )
# print( unknown )



def get_cli_opts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    for agg in argv :  # While there are arguments left to parse...
        if agg[0] == '-':  # Found a "-name value" pair.

            agg = agg[1:]

            if agg[0] == '-': # just in case there were '--' then that should also go 
                agg = agg[1:]

            opts[agg] =  ( argv[1] )   # Add key and value to the dictionary.
    return opts



if __name__ == '__main__':
    from sys import argv
    myargs = get_cli_opts(argv)
    # if '-i' in myargs:  # Example usage.
    #     print(myargs['-i'])
    print(myargs)



