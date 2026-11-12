"""Identity, storage and mutation signatures for inference-only caches."""
def version(value):
    try:
        return value._version
    except RuntimeError:
        return None


def signature(value):
    if version(value) is None:
        return None
    return (id(value), value.data_ptr(), version(value), tuple(value.shape), tuple(value.stride()),
            value.storage_offset(), value.dtype, value.device)
