import ray
import gym
import torch
import os
import time
import numpy as np
import numpy.random as rd
import datetime
from buffer import ReplayBuffer, ReplayBufferMP
from evaluate import RecordEpisode, RecordEvaluate, Evaluator

"""
Modify [ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)
by https://github.com/GyChou
"""
observation_dim = 0
action_dim = 0

class Arguments:
    def __init__(self, configs):
        self.configs = configs
        self.gpu_id = configs['gpu_id']  # choose the GPU for running. gpu_id is None means set it automatically
        # current work directory. cwd is None means set it automatically
        self.cwd = configs['cwd'] if 'cwd' in configs.keys() else None
        # current work directory with time.
        self.if_cwd_time = configs['if_cwd_time'] if 'cwd' in configs.keys() else False
        self.expconfig = configs['expconfig']
        # initialize random seed in self.init_before_training()

        self.random_seed = configs['random_seed']
        # id state_dim action_dim reward_dim target_reward horizon_step
        self.env = configs['env']
        # Deep Reinforcement Learning algorithm
        self.agent = configs['agent']
        self.agent['agent_name'] = self.agent['class_name']().__class__.__name__
        self.trainer = configs['trainer']
        self.interactor = configs['interactor']
        self.buffer = configs['buffer']
        self.evaluator = configs['evaluator']
        self.InitDict = configs['InitDict']

        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)

        '''if_per_explore'''
        if self.buffer['if_on_policy']:
            self.if_per_explore = False
        else:
            self.if_per_explore = True

    def init_before_training(self, if_main=True):
        '''set gpu_id automatically'''
        if self.gpu_id is None:  # set gpu_id automatically
            import sys
            self.gpu_id = sys.argv[-1][-4]
        else:
            self.gpu_id = self.gpu_id
        # if not self.gpu_id.isdigit():  # set gpu_id as '0' in default
        #     self.gpu_id = '0'

        '''set cwd automatically'''
        if self.cwd is None:
            if self.if_cwd_time:
                curr_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            else:
                curr_time = 'current'
            if self.expconfig is None:
                self.cwd = f'./logs/{self.env["id"]}/{self.agent["agent_name"]}/exp_{curr_time}_cuda:{self.gpu_id}'
            else:
                self.cwd = f'./logs/{self.env["id"]}/{self.agent["agent_name"]}/exp_{curr_time}_cuda:{self.gpu_id}_{self.expconfig}'

        if if_main:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)
            fw = open(self.cwd+"/config.txt",'w+') 
            fw.write(str(self.configs)) #把字典转化为str 
            fw.close()

        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        """Preprocess a standard OpenAI gym environment for training.

        `object env` a standard OpenAI gym environment, it has env.reset() and env.step()
        `bool if_print` print the information of environment. Such as env_name, state_dim ...
        `object data_type` convert state (sometimes float64) to data_type (float32).
        """
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)
        self.env.env_num = getattr(self.env, 'env_num', 1)
        self.env_num = 1

        state_avg, state_std = get_avg_std__for_state_norm(self.env_name)
        self.neg_state_avg = -state_avg
        self.div_state_std = 1 / (state_std + 1e-4)

        self.reset = self.reset_norm
        self.step = self.step_norm

    def reset_norm(self) -> np.ndarray:
        """ convert the data type of state from float64 to float32
        do normalization on state

        return `array state` state.shape==(state_dim, )
        """
        state = self.env.reset()
        state = (state + self.neg_state_avg) * self.div_state_std
        return state.astype(np.float32)

    def step_norm(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        """convert the data type of state from float64 to float32,
        adjust action range to (-action_max, +action_max)
        do normalization on state

        return `array state`  state.shape==(state_dim, )
        return `float reward` reward of one step
        return `bool done` the terminal of an training episode
        return `dict info` the information save in a dict. OpenAI gym standard. Send a `None` is OK
        """
        state, reward, done, info = self.env.step(action * self.action_max)
        state = (state + self.neg_state_avg) * self.div_state_std
        return state.astype(np.float32), reward, done, info

def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    """get information of a standard OpenAI gym env.

    The DRL algorithm AgentXXX need these env information for building networks and training.

    `object env` a standard OpenAI gym environment, it has env.reset() and env.step()
    `bool if_print` print the information of environment. Such as env_name, state_dim ...
    return `env_name` the environment name, such as XxxXxx-v0
    return `state_dim` the dimension of state
    return `action_dim` the dimension of continuous action; Or the number of discrete action
    return `action_max` the max action of continuous action; action_max == 1 when it is discrete action space
    return `max_step` the steps in an episode. (from env.reset to done). It breaks an episode when it reach max_step
    return `if_discrete` Is this env a discrete action space?
    return `target_return` the target episode return, if agent reach this score, then it pass this game (env).
    """
    assert isinstance(env, gym.Env)

    env_name = getattr(env, 'env_name', None)
    env_name = env.unwrapped.spec.id if env_name is None else env_name

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_return = getattr(env, 'target_return', None)
    target_return_default = getattr(env.spec, 'reward_threshold', None)
    if target_return is None:
        target_return = target_return_default
    if target_return is None:
        target_return = 2 ** 16

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return

def get_avg_std__for_state_norm(env_name) -> (np.ndarray, np.ndarray):
    """return the state normalization data: neg_avg and div_std

    ReplayBuffer.print_state_norm() will print `neg_avg` and `div_std`
    You can save these array to here. And PreprocessEnv will load them automatically.
    eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
    neg_avg = -states.mean()
    div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())

    `str env_name` the name of environment that helps to find neg_avg and div_std
    return `array avg` neg_avg.shape=(state_dim)
    return `array std` div_std.shape=(state_dim)
    """
    avg = 0
    std = 1
    # if env_name == 'LunarLanderContinuous-v2':
    #     avg = np.array([1.65470898e-02, -1.29684399e-01, 4.26883133e-03, -3.42124557e-02,
    #                     -7.39076972e-03, -7.67103031e-04, 1.12640885e+00, 1.12409466e+00])
    #     std = np.array([0.15094465, 0.29366297, 0.23490797, 0.25931464, 0.21603736,
    #                     0.25886878, 0.277233, 0.27771219])
    # elif env_name == "BipedalWalker-v3":
    #     avg = np.array([1.42211734e-01, -2.74547996e-03, 1.65104509e-01, -1.33418152e-02,
    #                     -2.43243194e-01, -1.73886203e-02, 4.24114229e-02, -6.57800099e-02,
    #                     4.53460692e-01, 6.08022244e-01, -8.64884810e-04, -2.08789053e-01,
    #                     -2.92092949e-02, 5.04791247e-01, 3.33571745e-01, 3.37325723e-01,
    #                     3.49106580e-01, 3.70363115e-01, 4.04074671e-01, 4.55838055e-01,
    #                     5.36685407e-01, 6.70771701e-01, 8.80356865e-01, 9.97987386e-01])
    #     std = np.array([0.84419678, 0.06317835, 0.16532085, 0.09356959, 0.486594,
    #                     0.55477525, 0.44076614, 0.85030824, 0.29159821, 0.48093035,
    #                     0.50323634, 0.48110776, 0.69684234, 0.29161077, 0.06962932,
    #                     0.0705558, 0.07322677, 0.07793258, 0.08624322, 0.09846895,
    #                     0.11752805, 0.14116005, 0.13839757, 0.07760469])
    # elif env_name == 'ReacherBulletEnv-v0':
    #     avg = np.array([0.03149641, 0.0485873, -0.04949671, -0.06938662, -0.14157104,
    #                     0.02433294, -0.09097818, 0.4405931, 0.10299437], dtype=np.float32)
    #     std = np.array([0.12277275, 0.1347579, 0.14567468, 0.14747661, 0.51311225,
    #                     0.5199606, 0.2710207, 0.48395795, 0.40876198], dtype=np.float32)
    # elif env_name == 'AntBulletEnv-v0':
    #     avg = np.array([-1.4400886e-01, -4.5074993e-01, 8.5741436e-01, 4.4249415e-01,
    #                     -3.1593361e-01, -3.4174921e-03, -6.1666980e-02, -4.3752361e-03,
    #                     -8.9226037e-02, 2.5108769e-03, -4.8667483e-02, 7.4835382e-03,
    #                     3.6160579e-01, 2.6877613e-03, 4.7474738e-02, -5.0628246e-03,
    #                     -2.5761038e-01, 5.9789192e-04, -2.1119279e-01, -6.6801407e-03,
    #                     2.5196713e-01, 1.6556121e-03, 1.0365561e-01, 1.0219718e-02,
    #                     5.8209229e-01, 7.7563477e-01, 4.8815918e-01, 4.2498779e-01],
    #                    dtype=np.float32)
    #     std = np.array([0.04128463, 0.19463477, 0.15422264, 0.16463493, 0.16640785,
    #                     0.08266512, 0.10606721, 0.07636797, 0.7229637, 0.52585346,
    #                     0.42947173, 0.20228386, 0.44787514, 0.33257666, 0.6440182,
    #                     0.38659114, 0.6644085, 0.5352245, 0.45194066, 0.20750992,
    #                     0.4599643, 0.3846344, 0.651452, 0.39733195, 0.49320385,
    #                     0.41713253, 0.49984455, 0.4943505], dtype=np.float32)
    # elif env_name == 'HumanoidBulletEnv-v0':
    #     avg = np.array([-1.25880212e-01, -8.51390958e-01, 7.07488894e-01, -5.72232604e-01,
    #                     -8.76260102e-01, -4.07587215e-02, 7.27005303e-04, 1.23370838e+00,
    #                     -3.68912554e+00, -4.75829793e-03, -7.42472351e-01, -8.94218776e-03,
    #                     1.29535913e+00, 3.16205365e-03, 9.13809776e-01, -6.42679911e-03,
    #                     8.90435696e-01, -7.92571157e-03, 6.54826105e-01, 1.82383414e-02,
    #                     1.20868635e+00, 2.90832808e-03, -9.96598601e-03, -1.87555347e-02,
    #                     1.66691601e+00, 7.45300390e-03, -5.63859344e-01, 5.48619963e-03,
    #                     1.33900166e+00, 1.05895223e-02, -8.30249667e-01, 1.57017610e-03,
    #                     1.92912612e-02, 1.55787319e-02, -1.19833803e+00, -8.22103582e-03,
    #                     -6.57119334e-01, -2.40323972e-02, -1.05282271e+00, -1.41856335e-02,
    #                     8.53593826e-01, -1.73063378e-03, 5.46878874e-01, 5.43514848e-01],
    #                    dtype=np.float32)
    #     std = np.array([0.08138401, 0.41358876, 0.33958328, 0.17817754, 0.17003846,
    #                     0.15247536, 0.690917, 0.481272, 0.40543965, 0.6078898,
    #                     0.46960834, 0.4825346, 0.38099176, 0.5156369, 0.6534775,
    #                     0.45825616, 0.38340876, 0.89671516, 0.14449312, 0.47643778,
    #                     0.21150663, 0.56597894, 0.56706554, 0.49014297, 0.30507362,
    #                     0.6868296, 0.25598812, 0.52973163, 0.14948095, 0.49912784,
    #                     0.42137524, 0.42925757, 0.39722264, 0.54846555, 0.5816031,
    #                     1.139402, 0.29807225, 0.27311933, 0.34721208, 0.38530213,
    #                     0.4897849, 1.0748593, 0.30166605, 0.30824476], dtype=np.float32)
    # elif env_name == 'MinitaurBulletEnv-v0': # need check
    #     avg = np.array([0.90172989, 1.54730119, 1.24560906, 1.97365306, 1.9413892,
    #                     1.03866835, 1.69646277, 1.18655352, -0.45842347, 0.17845232,
    #                     0.38784456, 0.58572877, 0.91414561, -0.45410697, 0.7591031,
    #                     -0.07008998, 3.43842258, 0.61032482, 0.86689961, -0.33910894,
    #                     0.47030415, 4.5623528, -2.39108079, 3.03559422, -0.36328256,
    #                     -0.20753499, -0.47758384, 0.86756409])
    #     std = np.array([0.34192648, 0.51169916, 0.39370621, 0.55568461, 0.46910769,
    #                     0.28387504, 0.51807949, 0.37723445, 13.16686185, 17.51240024,
    #                     14.80264211, 16.60461412, 15.72930229, 11.38926597, 15.40598346,
    #                     13.03124941, 2.47718145, 2.55088804, 2.35964651, 2.51025567,
    #                     2.66379017, 2.37224904, 2.55892521, 2.41716885, 0.07529733,
    #                     0.05903034, 0.1314812, 0.0221248])
    return avg, std



def make_env(env_dict, id=None):


    env = gym.make(env_dict['id'])
    #env = PreprocessEnv(env, if_print=False)
    # else:
    #     # create an environment for policy learning from low-dimensional observations
    #     env = gym.make(env_dict['id'])
    #     env = gym.wrappers.FlattenObservation(env)
    global observation_dim
    global action_dim
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]

    # env = DeltaActionEnvWrapper(env)
    # env = CurriculumWrapper(env,
    #                    intervention_actors=[GoalInterventionActorPolicy()],
    #                    actives=[(0, 1000000000, 1, 0)])
    
    return env
'''
POMDP envs
'''
#from causal_world.envs.causalworld import CausalWorld
#from causal_world.task_generators.task import generate_task
#from causal_world.wrappers.action_wrappers import DeltaActionEnvWrapper
#from causal_world.intervention_actors import GoalInterventionActorPolicy
#from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper

#class POMDPEnv(gym.Env):
#    def __init__(self,env_dict) -> None:
#        super().__init__()
#        task = generate_task(task_generator_id=env_dict['id'])
#        env = CausalWorld(task=task,action_mode=env_dict['action_mode'],skip_frame=10,max_episode_length=env_dict['max_step'])
#        self.env = CurriculumWrapper(env,
#                      intervention_actors=[GoalInterventionActorPolicy()],
#                      actives=[(0, 1000000000, 1, 0)])
#        self.observation_space = env.observation_space
#        self.action_space = env.action_space
#        self.next_state = np.zeros(env.observation_space.shape[0]-9)
#    def step(self, action):
#        temp_next_s, reward, done, info = self.env.step(action)
#        self.next_state[0:10] = temp_next_s[0:10]
#        self.next_state[10:] = temp_next_s[19:]
#        return self.next_state, reward, done, info
#    def reset(self):
#        state = self.env.reset()
#        self.next_state[0:10] = state[0:10]
#        self.next_state[10:] = state[19:]
#        return self.next_state

# def make_env(env_dict, id=None):
#     # task = generate_task(task_generator_id=env_dict['id'])
#     # env = CausalWorld(task=task,action_mode=env_dict['action_mode'],skip_frame=10,max_episode_length=env_dict['max_step'])
#     env = POMDPEnv(env_dict)
#     global observation_dim
#     global action_dim
#     observation_dim = env.observation_space.shape[0]-9
#     action_dim = env.action_space.shape[0]
#     # # env = DeltaActionEnvWrapper(env)
#     # env = CurriculumWrapper(env,
#     #                   intervention_actors=[GoalInterventionActorPolicy()],
#     #                   actives=[(0, 1000000000, 1, 0)])
    
#     return env


@ray.remote
class InterActor(object):

    def __init__(self, id, args):
        self.id = id
        args.init_before_training(if_main=False)
        self.env = make_env(args.env, self.id)
        self.env_max_step = args.env['max_step']
        global observation_dim
        global action_dim
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]    
        self.reward_scale = args.interactor['reward_scale']
        self._horizon_step = args.interactor['horizon_step'] // args.interactor['rollout_num']
        self.gamma = args.interactor['gamma'] if type(args.interactor['gamma']) is np.ndarray else np.ones(
            args.env['reward_dim']) * args.interactor['gamma']
        self.action_dim = action_dim
        self.if_discrete_action = args.env['if_discrete_action']
        if args.agent['agent_name'] in ['AgentPPO']:
            self.modify_action = lambda x: np.tanh(x)
        else:
            self.modify_action = lambda x: x
        self.buffer = ReplayBuffer(
            max_len=args.buffer['max_buf'] // args.interactor['rollout_num'] + args.env['max_step'],
            block_size=args.InitDict['block_size'],
            if_on_policy=args.buffer['if_on_policy'],
            state_dim=observation_dim,
            action_dim=1 if self.if_discrete_action else action_dim,
            reward_dim=args.env['reward_dim'],
            if_per=False,
            if_gpu=False)
        self.block_size = args.InitDict['block_size']
        if args.InitDict['use_attbias']:
            self.att_bias = torch.ones([args.InitDict[ 'block_size_state'],args.InitDict[ 'block_size'],observation_dim])
            for i in range(args.InitDict[ 'block_size_state']):
                for j in range(args.InitDict[ 'block_size']):
                    if j == 0:
                        self.att_bias[i,j,:] = args.agent['gamma_att']
                    else:
                        self.att_bias[i,j,:] = self.att_bias[i,j-1,:] *  args.agent['gamma_att']
            # #Joint Positions
            # for i in range(1,9+1):
            #     self.att_bias[:,:,i] = 1
            # #End-effector Positions
            # for i in range(10,18+1):
            #     self.att_bias[:,:,i] = 1
            
            # for i in range(23,35+1):
            #     self.att_bias[:,:,i] = 1

        else:
            self.att_bias = None

        self.record_episode = RecordEpisode()

    @ray.method(num_returns=1)
    def explore_env(self, select_action, policy,device):
        self.buffer.empty_buffer_before_explore()
        actual_step = 0
        actual_traj = 0
        while actual_step < self._horizon_step:
            state = self.env.reset()
            state_past = np.zeros([self.block_size,state.size])
            for i in range(self.env_max_step):

                state_past[:-1] = state_past[1:]
                state_past[-1] = state

                action = select_action(state,state_past, policy, self.att_bias,device)
                next_s, reward, done, _ = self.env.step(self.modify_action(action))
                done = True if i == (self.env_max_step - 1) else done
                self.buffer.append_buffer(state,
                                          state_past,
                                          action,
                                          reward * self.reward_scale,
                                          np.zeros(self.gamma.shape) if done else self.gamma)
                actual_step += 1
                if done:
                    actual_traj+=1
                    break
                state = next_s
        self.buffer.update_now_len_before_sample()
        return actual_traj, \
               self.buffer.buf_state[:self.buffer.now_len], \
               self.buffer.buf_state_past[:self.buffer.now_len], \
               self.buffer.buf_action[:self.buffer.now_len], \
               self.buffer.buf_reward[:self.buffer.now_len], \
               self.buffer.buf_gamma[:self.buffer.now_len]

    @ray.method(num_returns=1)
    def random_explore_env(self, r_horizon_step=None):
        self.buffer.empty_buffer_before_explore()
        if r_horizon_step is None:
            r_horizon_step = self._horizon_step
        else:
            r_horizon_step = max(min(r_horizon_step, self.buffer.max_len - 1), self._horizon_step)
        actual_traj = 0
        actual_step = 0
        while actual_step < r_horizon_step:
            state = self.env.reset()
            state_past = np.zeros([self.block_size,state.size])
            for _ in range(self.env_max_step):
                state_past[:-1] = state_past[1:]
                state_past[-1] = state
                action = rd.randint(self.action_dim) if self.if_discrete_action else rd.uniform(-1, 1,
                                                                                                size=self.action_dim)
                next_s, reward, done, _ = self.env.step(self.modify_action(action))
                self.buffer.append_buffer(state,
                                          state_past,
                                          action,
                                          reward * self.reward_scale,
                                          np.zeros(self.gamma.shape) if done else self.gamma)
                actual_step += 1
                if done:
                    actual_traj+=1
                    break
                state = next_s
        self.buffer.update_now_len_before_sample()
        return self.buffer.buf_state[:self.buffer.now_len], \
               self.buffer.buf_state_past[:self.buffer.now_len], \
               self.buffer.buf_action[:self.buffer.now_len], \
               self.buffer.buf_reward[:self.buffer.now_len], \
               self.buffer.buf_gamma[:self.buffer.now_len]

    def exploite_env(self, policy, eval_times,device):
        self.record_episode.clear()
        eval_record = RecordEvaluate()
        if self.att_bias is not None:
            att_bias = self.att_bias.to(device)
        else:
            att_bias = self.att_bias
        for _ in range(eval_times):
            state = self.env.reset()
            state_past = np.zeros([self.block_size,state.size])
            for _ in range(self.env_max_step):
                state_past[:-1] = state_past[1:]
                state_past[-1] = state
                action = policy(
                    torch.as_tensor((state,), dtype=torch.float32).detach_().to(device),
                    torch.as_tensor((state_past,), dtype=torch.float32).detach_().to(device),
                    att_bias,
                    )
                next_s, reward, done, info = self.env.step(action.detach().cpu().numpy()[0])
                self.record_episode.add_record(reward, info)
                if done:
                    break
                state = next_s
            eval_record.add(self.record_episode.get_result())
            self.record_episode.clear()
        return eval_record.results


class Trainer(object):

    def __init__(self, args_trainer, agent, buffer):
        self.agent = agent
        self.buffer = buffer
        self.sample_step = args_trainer['sample_step']
        self.batch_size = args_trainer['batch_size']
        self.policy_reuse = args_trainer['policy_reuse']

    def train(self):
        self.agent.act.to(device=self.agent.device)
        self.agent.cri.to(device=self.agent.device)
        train_record = self.agent.update_net(self.buffer, self.sample_step, self.batch_size, self.policy_reuse)
        if self.buffer.if_on_policy:
            self.buffer.empty_buffer_before_explore()
        return train_record


def beginer(config, params=None):
    args = Arguments(config)
    args.init_before_training()
    args_id = ray.put(args)
    #######Init######

    interactors = [InterActor.remote(i, args_id) for i in range(args.interactor['rollout_num'])]
    print('state dim',observation_dim)
    make_env(args.env)
    args.InitDict['state_dim'] = observation_dim
    args.InitDict['action_dim'] = action_dim
    print('state dim',observation_dim)
    agent = args.agent['class_name'](args.agent)
    agent.init(InitDict=args.InitDict,
               reward_dim=args.env['reward_dim'],
               if_per=args.buffer['if_per'],
               if_load_model=args.agent['if_load_model'],
               actor_path=args.agent['actor_path'],
               critic_path=args.agent['critic_path'])
    buffer_mp = ReplayBufferMP(
        max_len=args.buffer['max_buf'] + args.env['max_step'] * args.interactor['rollout_num'],
        block_size=args.InitDict['block_size'],
        state_dim=observation_dim,
        action_dim=1 if args.env['if_discrete_action'] else action_dim,
        reward_dim=args.env['reward_dim'],
        if_on_policy=args.buffer['if_on_policy'],
        if_per=args.buffer['if_per'],
        rollout_num=args.interactor['rollout_num'])
    trainer = Trainer(args.trainer, agent, buffer_mp)
    evaluator = Evaluator(args)
    rollout_num = args.interactor['rollout_num']

    #######Random Explore Before Interacting#######
    if args.if_per_explore:
        episodes_ids = [interactors[i].random_explore_env.remote() for i in range(rollout_num)]
        assert len(episodes_ids) > 0
        for i in range(len(episodes_ids)):
            done_id, episodes_ids = ray.wait(episodes_ids)
            buf_state,buf_state_past, buf_action, buf_reward, buf_gamma = ray.get(done_id[0])
            buffer_mp.extend_buffer(buf_state,buf_state_past, buf_action, buf_reward, buf_gamma, i)

    #######Interacting Begining#######
    start_time = time.time()
    device = 'cpu'
    policy_id = ray.put(agent.act.to(device))
    
    while (evaluator.record_totalstep < evaluator.break_step) or (evaluator.record_satisfy_reward):
        #######Explore Environment#######
        episodes_ids = [interactors[i].explore_env.remote(agent.select_action, policy_id,device) for i in
                        range(rollout_num)]
        assert len(episodes_ids) > 0
        sample_step = 0
        for i in range(len(episodes_ids)):
            done_id, episodes_ids = ray.wait(episodes_ids)
            actual_step, buf_state,buf_state_past, buf_action, buf_reward, buf_gamma = ray.get(done_id[0])
            sample_step += actual_step
            buffer_mp.extend_buffer(buf_state,buf_state_past, buf_action, buf_reward, buf_gamma, i)
        evaluator.update_totalstep(sample_step)
        #######Training#######
        trian_record = trainer.train()
        evaluator.tb_train(trian_record)
        #######Evaluate#######
        device = 'cpu'
        policy_id = ray.put(agent.act.to(device))
        evalRecorder = RecordEvaluate()
        if_eval = True
        #######pre-eval#######
        
        if evaluator.pre_eval_times > 0:
            eval_results = ray.get(
                [interactors[i].exploite_env.remote(policy_id, eval_times=evaluator.pre_eval_times, device=device) for i in
                 range(rollout_num)])
            for eval_result in eval_results:
                evalRecorder.add_many(eval_result)
            eval_record = evalRecorder.eval_result()
            if eval_record['reward'][0]['max'] < evaluator.target_reward:
                if_eval = False
                evaluator.tb_eval(eval_record)
        #######eval#######
        if if_eval:
            eval_results = ray.get(
                [interactors[i].exploite_env.remote(policy_id, eval_times=(evaluator.eval_times),device=device)
                 for i in range(rollout_num)])
            for eval_result in eval_results:
                evalRecorder.add_many(eval_result)
            eval_record = evalRecorder.eval_result()
            evaluator.tb_eval(eval_record)
        #######Save Model#######
        evaluator.analyze_result(eval_record)
        evaluator.iter_print(trian_record, eval_record, use_time=(time.time() - start_time))
        evaluator.save_model(agent.act, agent.cri)
        start_time = time.time()

    print(f'#######Experiment Finished!\t TotalTime:{evaluator.total_time:8.0f}s #######')
