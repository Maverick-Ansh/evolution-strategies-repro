"""Persistent XLA compilation cache, shared across every run in a sweep.

Measured on this box: a single Brax `generalized` rollout graph takes ~150 s of
SINGLE-THREADED CPU to compile (confirmed by sampling /proc/<pid>/stat: the process sits
at exactly 1.00 cores with both GPUs at 0% utilisation). An ES run compiles two or three
such graphs -- one per episode-cap bucket plus the evaluation rollout -- so roughly five
minutes of every run was compilation rather than simulation.

Across a sweep those graphs are almost all identical: the HLO depends on the environment,
the population size and the scan length, none of which change with the random seed. So a
persistent cache turns 12 ES runs' worth of compilation into about six distinct compiles.
On a 4-CPU box where compilation, not the GPU, is the bottleneck, this is the single
biggest speedup available -- and it changes nothing about the numerics.
"""
import os


def enable(path: str = None):
    import jax
    path = path or os.environ.get('JAX_CACHE_DIR', '/kaggle/working/jax_cache')
    os.makedirs(path, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', path)
    # cache everything that took more than a second to build, regardless of size
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)
    try:
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    except Exception:
        pass          # older JAX builds do not expose this knob
    return path
