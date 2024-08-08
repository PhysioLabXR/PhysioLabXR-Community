def rpc(func):
    """this is a decorator for rpc methods defined in RenaScripts"""
    func.is_rpc_method = True
    return func

def async_rpc(func):
    """this is a decorator for async rpc methods defined in RenaScripts"""
    func.is_async_rpc_method = True
    return func