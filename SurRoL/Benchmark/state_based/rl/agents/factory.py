from .ddpg import DDPG
from .ddpgbc import DDPGBC
from .dex import DEX
from .sac import SAC

AGENTS = {
    'DDPG': DDPG,
    'DDPGBC': DDPGBC,
    'DEX': DEX,
    'SAC': SAC,
}


def make_agent(env_params, sampler, cfg):
    if cfg.name not in AGENTS.keys():
        assert 'Agent is not supported: %s' % cfg.name
    else:
        return AGENTS[cfg.name](
            env_params=env_params,
            sampler=sampler,
            agent_cfg=cfg
        )
