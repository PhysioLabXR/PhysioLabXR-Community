import msgpack as mp
import msgpack_numpy as mp_np

# convert any py obj to bytes format
def to_bytes(py_obj) -> bytes:
    return mp.packb(py_obj, use_bin_type=True)

def from_bytes(b: bytes):
    return mp.unpackb(b, raw=False)