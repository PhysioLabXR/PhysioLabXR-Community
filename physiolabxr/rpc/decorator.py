def rpc(func):
    """
    this is a decorator for rpc methods defined in RenaScripts
    """
    func.is_rpc_method = True
    return func