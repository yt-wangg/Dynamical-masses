"""JAX helper utilities.

This module isolates JAX setup logic so model modules can stay focused.
"""

import os

def _configure_jax_gpu_fallback():
    """
    Configure JAX to prefer GPU but fall back to CPU if GPU is not available.

    Returns
    -------
    jax : module
        The configured JAX module
    """
    import warnings
    import sys
    import importlib

    # Try GPU first, but fall back to CPU if not available
    try:
        # Clear any existing JAX platform preference to allow auto-detection
        if 'JAX_PLATFORMS' in os.environ:
            del os.environ['JAX_PLATFORMS']

        # Try to import jax with auto-detection
        import jax
        devices = jax.devices()

        # If no GPU devices are available, switch to CPU
        if not any(device.platform == 'gpu' for device in devices):
            warnings.warn("No GPU devices available, switching to CPU backend")
            os.environ['JAX_PLATFORMS'] = 'cpu'
            # Need to reload modules to pick up the platform change
            importlib.reload(jax)

    except Exception as e:
        warnings.warn(f"Failed to initialize JAX GPU backend: {e}. Using CPU backend")
        os.environ['JAX_PLATFORMS'] = 'cpu'
        # Clear any cached imports and retry with CPU
        modules_to_remove = [mod for mod in sys.modules.keys() if mod.startswith('jax')]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]
        import jax

    return jax



