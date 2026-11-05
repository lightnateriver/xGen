"""Translate public H3 Base denoising steps at the Diffusers boundary."""


def schedule_points(steps, cache_file=None):
    if type(steps) is not int or steps < 1:
        raise ValueError("H3 num_inference_steps must be a positive integer")
    if cache_file and steps != cache_steps(cache_file):
        raise ValueError("H3 num_inference_steps must match the Base AdaLN cache evaluation count")
    # MiniMaxH3Scheduler expects the terminal sigma=0 in its point count.
    return steps + 1


def cache_steps(path):
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(path, framework="pt") as cache:
            metadata = cache.metadata() or {}
        points = int(metadata["num_steps"])
        steps = int(metadata.get("num_eval_steps", points - 1))
    except (OSError, SafetensorError, KeyError, ValueError) as exc:
        raise ValueError("Invalid H3 AdaLN cache schedule metadata") from exc
    if steps < 1 or points != steps + 1:
        raise ValueError("H3 AdaLN cache must contain one more sigma point than evaluations")
    return steps
