"""Detached sweep launcher: N worker processes across the available GPUs.

Two hard-won constraints shape this file.

1. NEVER BLOCK THE NOTEBOOK KERNEL. A long-running foreground cell driven over MCP has
   no interrupt and eventually trips the client's idle timeout, taking the kernel with
   it. So every run is a detached process (`start_new_session=True`) writing to its own
   log file, and progress is read by tailing those files from short cells.

2. RUNS MUST BE RESUMABLE. Each run writes results/<tag>.json and skips itself if that
   file already exists, so a killed sweep resumes instead of restarting.

A single ES run is launch-bound rather than compute-bound on a T4 -- the rollout is a
sequential scan of small batched steps, so the GPU idles between kernels. Packing
several processes onto one GPU therefore buys real aggregate throughput here, which is
the opposite of what happens for compute-bound training. `--per-gpu` controls it, and
XLA_PYTHON_CLIENT_MEM_FRACTION keeps them from fighting over memory.
"""
import argparse, itertools, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE1 = ['halfcheetah', 'hopper', 'inverted_double_pendulum',
          'inverted_pendulum', 'swimmer', 'walker2d']


def es_cmd(env, seed, steps, tag, **kw):
    c = [sys.executable, 'scripts/train_es.py', '--env', env, '--seed', str(seed),
         '--max-steps', str(steps), '--tag', tag]
    for k, v in kw.items():
        flag = '--' + k.replace('_', '-')
        c += [flag] if v is True else [flag, str(v)]
    return c


def pg_cmd(env, seed, steps, tag, **kw):
    c = [sys.executable, 'scripts/train_ppo.py', '--env', env, '--seed', str(seed),
         '--max-steps', str(steps), '--tag', tag]
    for k, v in kw.items():
        flag = '--' + k.replace('_', '-')
        c += [flag] if v is True else [flag, str(v)]
    return c


def build_runs(which, seeds, es_steps, pg_steps, npairs):
    runs = []
    if 'table1' in which:
        for env, s in itertools.product(TABLE1, range(seeds)):
            runs.append((es_cmd(env, s, es_steps, 'es_{}_s{}'.format(env, s),
                                n_pairs=npairs), 'es_{}_s{}'.format(env, s)))
            runs.append((pg_cmd(env, s, pg_steps, 'ppo_{}_s{}'.format(env, s)),
                         'ppo_{}_s{}'.format(env, s)))
    if 'frameskip' in which:
        # Sec. 4.4: "running the Atari game Pong using a frame skip parameter in {1,2,3,4}"
        for env in ['hopper', 'walker2d']:
            for k, s in itertools.product([1, 2, 3, 4], range(seeds)):
                runs.append((es_cmd(env, s, es_steps, 'fsES_{}_k{}_s{}'.format(env, k, s),
                                    n_pairs=npairs, action_repeat=k),
                             'fsES_{}_k{}_s{}'.format(env, k, s)))
                runs.append((pg_cmd(env, s, pg_steps, 'fsPG_{}_k{}_s{}'.format(env, k, s),
                                    action_repeat=k),
                             'fsPG_{}_k{}_s{}'.format(env, k, s)))
    if 'delayed' in which:
        # Sec. 3.3: "ES can deal with maximally sparse and delayed rewards"
        for env in ['hopper', 'halfcheetah']:
            for s in range(seeds):
                runs.append((es_cmd(env, s, es_steps, 'dlES_{}_s{}'.format(env, s),
                                    n_pairs=npairs, delayed_reward=True),
                             'dlES_{}_s{}'.format(env, s)))
                runs.append((pg_cmd(env, s, pg_steps, 'dlPG_{}_s{}'.format(env, s),
                                    delayed_reward=True),
                             'dlPG_{}_s{}'.format(env, s)))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--which', default='table1')
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--es-steps', type=int, default=20_000_000)
    p.add_argument('--pg-steps', type=int, default=5_000_000)
    p.add_argument('--n-pairs', type=int, default=128)
    p.add_argument('--gpus', type=int, default=2)
    p.add_argument('--per-gpu', type=int, default=2)
    p.add_argument('--mem-fraction', type=float, default=0.42)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    os.chdir(ROOT)
    os.makedirs('results/logs', exist_ok=True)
    runs = build_runs(a.which.split(','), a.seeds, a.es_steps, a.pg_steps, a.n_pairs)
    todo = [(c, t) for c, t in runs if not os.path.exists('results/{}.json'.format(t))]
    print("{} runs, {} still to do".format(len(runs), len(todo)), flush=True)
    if a.dry_run:
        for c, t in todo:
            print(' ', t, ' '.join(c[1:]))
        return

    slots = a.gpus * a.per_gpu
    running, queue, done = [], list(todo), 0
    t0 = time.time()
    while queue or running:
        while queue and len(running) < slots:
            cmd, tag = queue.pop(0)
            gpu = len(running) % a.gpus
            env = dict(os.environ,
                       CUDA_VISIBLE_DEVICES=str(gpu),
                       XLA_PYTHON_CLIENT_PREALLOCATE='false',
                       XLA_PYTHON_CLIENT_MEM_FRACTION=str(a.mem_fraction),
                       MUJOCO_GL='egl')
            log = open('results/logs/{}.log'.format(tag), 'w')
            pr = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=ROOT, env=env, start_new_session=True)
            running.append((pr, tag, log, time.time(), gpu))
            print("[launch gpu{}] {}".format(gpu, tag), flush=True)
        time.sleep(5)
        for item in list(running):
            pr, tag, log, ts, gpu = item
            if pr.poll() is not None:
                log.close()
                running.remove(item)
                done += 1
                ok = os.path.exists('results/{}.json'.format(tag))
                print("[done {}/{}] {} rc={} {:.0f}s {}".format(
                    done, len(todo), tag, pr.returncode, time.time() - ts,
                    "OK" if ok else "NO OUTPUT"), flush=True)
    print("sweep finished in {:.0f}s".format(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
