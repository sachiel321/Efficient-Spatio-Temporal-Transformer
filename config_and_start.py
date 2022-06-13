from interaction import beginer
from agent import *
import ray

max_step = 300
rollout_num = 4
config_ppo = {
    'gpu_id': 0,
    'cwd': None,
    'if_cwd_time': True,
    'expconfig': None,
    'random_seed': 0,
    'env': {
        'id': 'HopperPyBulletEnv-v0',
        'state_dim': 15,
        'action_dim': 2,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 0,
        'max_step': max_step,
    },
    'agent': {
        'if_load_model':False,
        'actor_path':None,
        'critic_path':None,
        'class_name': AgentPPO,
        'net_dim': 128,
        'ratio_clip': 0.3,
        'lambda_entropy': 0.04,
        'lambda_gae_adv': 0.97,
        'if_use_gae': True,
        'if_use_dn': False,
        'learning_rate': 1e-4,
        'soft_update_tau': 2 ** -8,
        'gamma_att' : 0.85,
    },
    'trainer': {
        'batch_size': 256,
        'policy_reuse': 2 ** 4,
        'sample_step': max_step * rollout_num,
    },
    'interactor': {
        'horizon_step': max_step * rollout_num,
        'reward_scale': 2 ** 0,
        'gamma': 0.99,
        'rollout_num': rollout_num,
    },
    'buffer': {
        'max_buf': max_step * rollout_num,
        'if_on_policy': True,
        'if_per': False,
    },
    'evaluator': {
        'pre_eval_times': 2,  # for every rollout_worker 0 means cencle pre_eval
        'eval_times': 4,  # for every rollout_worker
        'if_save_model': True,
        'break_step': 2e6,
        'satisfy_reward_stop': False,
    },
    'InitDict':{
        'state_dim': None,
        'mid_dim':128,
        'embeddingT' : 32,
        'embeddingS':32,
        'atthead': 4,
        'attlayer': 1,
        'action_dim': None,
        'block_size': 6,
        'block_size_state':1,
        'batch_size': 256,
        'use_TS': False,
        'use_GTrXL':True,
        'use_attbias':False,
        'init_gru_gate_bias':2.0
    },
}


def demo2_ppo():

    env = {
        'id': 'LunarLanderContinuous-v2',
        'state_dim': 37,
        'action_dim': 9,
        'if_discrete_action': False,
        'reward_dim': 1,
        'target_reward': 2e6,
        'max_step': 200,
        #'action_mode': 'joint_positions'
    }
    config_ppo['InitDict']['state_dim'] = env['state_dim']
    config_ppo['InitDict']['action_dim'] = env['action_dim']
    config_ppo['InitDict']['block_size'] = 9
    config_ppo['agent']['if_load_model'] = False
    config_ppo['agent']['actor_path'] = ''
    config_ppo['agent']['critic_path'] = ''
    config_ppo['agent']['lambda_entropy'] = 0.05
    config_ppo['agent']['lambda_gae_adv'] = 0.97
    config_ppo['interactor']['rollout_num'] = 16
    config_ppo['agent']['learning_rate'] = 1e-4
    config_ppo['trainer']['batch_size'] = 1024
    config_ppo['trainer']['sample_step'] = env['max_step'] * config_ppo['interactor']['rollout_num']
    config_ppo['InitDict']['batch_size'] = config_ppo['trainer']['batch_size']
    config_ppo['interactor']['horizon_step'] = config_ppo['trainer']['sample_step']
    config_ppo['trainer']['policy_reuse'] = 4
    config_ppo['interactor']['gamma'] = 0.99
    config_ppo['evaluator']['break_step'] = int(1e5)
    config_ppo['buffer']['max_buf'] = config_ppo['interactor']['horizon_step']
    config_ppo['env'] = env
    config_ppo['gpu_id'] = '2'
    config_ppo['if_cwd_time'] = True
    config_ppo['expconfig'] = 'TS'
    if config_ppo['InitDict']['use_GTrXL'] and config_ppo['InitDict']['use_attbias']:
        config_ppo['expconfig'] = config_ppo['expconfig']+'attbias_GTrXL'
    elif config_ppo['InitDict']['use_GTrXL']:
        config_ppo['expconfig'] = config_ppo['expconfig']+'GTrXL'
    elif config_ppo['InitDict']['use_attbias']:
        config_ppo['expconfig'] = config_ppo['expconfig']+'attbias'
    else:
        config_ppo['expconfig'] = config_ppo['expconfig']+'MLP'
    config_ppo['random_seed'] = 49
    beginer(config_ppo)



if __name__ == '__main__':
    ray.init()
    # ray.init(local_mode=True)
    #ray.init(num_cpus=32)
    # ray.init(num_cpus=12, num_gpus=0)

    #demo1_sac()
    demo2_ppo()
    # demo_carla_sac()
    # demo_carla_ppo()
