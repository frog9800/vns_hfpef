"""
Shared DreamerV3 wiring for the VNS pipeline.

DropletRunner duplicates this block in data_collector.py, policy_runner.py
and train_offline.py. Factored out here so the observation space, the action
space and the encoder/decoder key regexes cannot drift apart between data
collection, training and deployment. If they drift, the checkpoint silently
loads into a different model and the policy misbehaves in ways that look
like a control problem rather than a plumbing problem.

THE ASYMMETRY THAT MATTERS
    encoder.mlp_keys = 'vector'            <- HR, MAP, refs, prev action, T
    decoder.mlp_keys = 'vector'            <- default
                     = 'vector|aux'        <- with decode_aux=True
'aux' carries filling pressure. It may appear in the DECODER only. The
moment it appears in encoder.mlp_keys the trained policy needs a filling
pressure sensor at deployment, and no such chronic implantable sensor
exists. There is no runtime check that catches this, so it lives here.
"""

import os
import sys
import numpy as np

DEFAULT_DREAMER_PATH = os.environ.get(
    "DREAMERV3_PATH", os.path.expanduser("~/cyberrunner/dreamerv3"))


def add_dreamer_to_path(path=None):
    path = path or DEFAULT_DREAMER_PATH
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"DreamerV3 not found at {path}. Clone the DropletRunner fork "
            f"and either put it there or set DREAMERV3_PATH.")
    sys.path.insert(0, path)
    sys.path.insert(0, os.path.join(path, "dreamerv3"))


def spaces(obs_dim, act_dim, aux_dim=None):
    import embodied
    obs_space = {
        "vector": embodied.Space(np.float32, (obs_dim,)),
        "reward": embodied.Space(np.float32),
        "is_first": embodied.Space(bool),
        "is_last": embodied.Space(bool),
        "is_terminal": embodied.Space(bool),
    }
    if aux_dim:
        obs_space["aux"] = embodied.Space(np.float32, (aux_dim,))
    act_space = {
        "action": embodied.Space(np.float32, (act_dim,), -1.0, 1.0),
        "reset": embodied.Space(bool),
    }
    return obs_space, act_space


def make_config(logdir, obs_dim, act_dim, aux_dim=None, batch_size=16,
                batch_length=64, size="small", extra=None):
    import embodied
    from dreamerv3 import agent as agt

    decoder_mlp = "vector|aux" if aux_dim else "vector"
    config = embodied.Config(agt.Agent.configs["defaults"])
    config = config.update(agt.Agent.configs[size])
    config = config.update({
        "logdir": str(logdir),
        "task": "vns_hfpef",
        "replay": "uniform",
        "replay_size": int(1e6),
        "replay_online": False,
        "batch_size": batch_size,
        "batch_length": batch_length,
        "run.train_ratio": 128,
        "run.log_every": 60,
        "run.save_every": 20,
        # no camera in this plant: the CNN branch is switched off with a
        # regex that matches nothing.
        "encoder.cnn_keys": "$^",
        "decoder.cnn_keys": "$^",
        "encoder.mlp_keys": "vector",
        "decoder.mlp_keys": decoder_mlp,
    })
    if extra:
        config = config.update(extra)
    return config


def load_agent(policy_dir, obs_dim, act_dim, aux_dim=None, cpu=True,
               dreamer_path=None):
    """
    Build an agent and restore a checkpoint. Returns (agent, step_counter).
    Raises FileNotFoundError if there is no checkpoint.
    """
    if cpu:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    add_dreamer_to_path(dreamer_path)

    import embodied
    from dreamerv3 import agent as agt

    ckpt_path = os.path.join(policy_dir, "checkpoint.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")

    obs_space, act_space = spaces(obs_dim, act_dim, aux_dim)
    config = make_config(policy_dir, obs_dim, act_dim, aux_dim,
                         batch_size=1, batch_length=1,
                         extra={"jax.platform": "cpu"} if cpu else None)

    step = embodied.Counter()
    agent = agt.Agent(obs_space, act_space, step, config)

    replay_dir = os.path.join(policy_dir, "replay")
    os.makedirs(replay_dir, exist_ok=True)
    replay = embodied.replay.Uniform(length=config.batch_length,
                                     capacity=int(1e4), directory=replay_dir)
    ckpt = embodied.Checkpoint(ckpt_path)
    ckpt.step = step
    ckpt.agent = agent
    ckpt.replay = replay
    ckpt.load_or_save()
    return agent, step


class AgentPolicy:
    """Thin stateful wrapper so callers do not manage agent_state by hand."""

    def __init__(self, agent, aux_dim=None, mode="eval"):
        self.agent = agent
        self.aux_dim = aux_dim
        self.mode = mode
        self.state = None

    def reset(self):
        self.state = None

    def __call__(self, vector, reward=0.0, is_first=False, is_terminal=False):
        obs = {
            "vector": np.asarray(vector, np.float32)[None],
            "reward": np.array([reward], np.float32),
            "is_first": np.array([bool(is_first)]),
            "is_last": np.array([False]),
            "is_terminal": np.array([bool(is_terminal)]),
        }
        if self.aux_dim:
            # present so the decoder head has its key; never encoded.
            obs["aux"] = np.zeros((1, self.aux_dim), np.float32)
        act, self.state = self.agent.policy(obs, self.state, mode=self.mode)
        return np.clip(np.asarray(act["action"][0], np.float32), -1.0, 1.0)
