def rpc(func):
    """
    this is a decorator for rpc methods defined in RenaScripts
    """
    def wrapper(*args, **kwargs):
        # Pre-invocation logic here (e.g., logging, validation)
        result = func(*args, **kwargs)
        # Post-invocation logic here
        return result
    return wrapper