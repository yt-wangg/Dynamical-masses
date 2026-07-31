"""JAX helper utilities.

This module isolates JAX setup logic so model modules can stay focused.
"""


def _configure_jax_gpu_fallback():
    """
    Use JAX's automatic device selection.

    JAX uses an available accelerator by default and falls back to CPU when
    no supported accelerator is available.

    Returns
    -------
    jax : module
        The configured JAX module
    """
    import jax

    return jax
