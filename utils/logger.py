import datetime

class Level(object):
    '''
    Level object.
    
    Inputs
    ------
    level : str
    Level.
    '''
    
    def __init__(
        self,
        level: str = "NOTSET"
    ):
        self._enum = {
            "CRITICAL": 50,
            "ERROR": 40,
            "WARNING": 30,
            "INFO": 20,
            "DEBUG": 10,
            "NOTSET": 0,
        }
        self._level = level
        self._val = self._enum[self._level]
        self._str = level

class Logger(object):
    '''
    Light logger class object.
    
    Inputs
    ------
    handler_name : str
    A name to distinguish between different logger objects.
    
    print_name : bool
    Whether or not to print handler_name.
    '''
    
    def __init__(
        self,
        handler_name: str,
        print_name: bool = True,
        level = "info",
        level_shorthand = False
    ):
        self._hname = handler_name
        self._print_name = print_name
        self._level = Level(level.upper())
    
    def _print_msg(
        self,
        obj_level,
        str_main,
        *args,
    ):
        _str_time = \
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _str_name = \
            (" [{}]".format(self._hname) if self._print_name else "")
        
        _str_arg = []
        for _iter in args:
            _str_arg.append(_iter)
        _str_arg = tuple(_str_arg)
        
        if obj_level._val >= self._level._val:
            print(
                "{}{} [{}] {}"\
                .format(
                    _str_time,
                    _str_name,
                    obj_level._level,
                    str_main%(_str_arg)
                )
            )
        
    def debug(
        self,
        str_main,
        *args
    ):
        self._print_msg(Level("DEBUG"), str_main, *args)
        
    def info(
        self,
        str_main,
        *args
    ):
        self._print_msg(Level("INFO"), str_main, *args)
        
    def warning(
        self,
        str_main,
        *args
    ):
        self._print_msg(Level("WARNING"), str_main, *args)
        
    def error(
        self,
        str_main,
        *args
    ):
        self._print_msg(Level("ERROR"), str_main, *args)
        
    def critical(
        self,
        str_main,
        *args
    ):
        self._print_msg(Level("CRITICAL"), str_main, *args)
