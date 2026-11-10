def supports_chunked_vae_decode(value):
    return callable(getattr(value, "decode_with_chunks", None)) and hasattr(value, "chunk_value_range")
