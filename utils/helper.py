from multiprocessing.pool import ThreadPool

def pool_this(n_processes = 8):
    """
    Light helper decorator to multi-processing/threading-ize any function with one arg.
    
    Credit: Nathan
    """    
    def my_decorator(func):
        def wrapped(func_arg_ls):
            with ThreadPool(n_processes) as p:
                result = p.map(func, func_arg_ls)
                p.close()
                p.join()
            return result
        return wrapped
    return my_decorator


