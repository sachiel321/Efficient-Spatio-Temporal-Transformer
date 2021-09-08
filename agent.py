import os
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from net import ActorPPOAtt
from net import CriticAdvAtt
from scipy.optimize import minimize
import scipy.signal
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO)  # 设置日志级别


class AgentBase:
    def __init__(self, args=None):
        self.learning_rate = 1e-4 if args is None else args['learning_rate']
        self.soft_update_tau = 2 ** -8 if args is None else args['soft_update_tau']  # 5e-3 ~= 2 ** -8
        self.state = None  # set for self.update_buffer(), initialize before training
        self.device = None

        self.act = self.act_target = None
        self.cri = self.cri_target = None
        self.act_optimizer = None
        self.cri_optimizer = None
        self.criterion = None
        self.get_obj_critic = None
        self.train_record = {}

    def init(self, net_dim, state_dim, action_dim, if_per=False):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        `int net_dim` the dimension of networks (the width of neural networks)
        `int state_dim` the dimension of state (the number of state vector)
        `int action_dim` the dimension of action (the number of discrete action)
        `bool if_per` Prioritized Experience Replay for sparse reward
        """

    def select_action(self, state) -> np.ndarray:
        """Select actions for exploration

        :array state: state.shape==(state_dim, )
        :return array action: action.shape==(action_dim, ), (action.min(), action.max())==(-1, +1)
        """
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device).detach_()
        action = self.act(states)[0]
        return action.cpu().numpy()

    def explore_env(self, env, buffer, target_step, reward_scale, gamma) -> int:
        """actor explores in env, then stores the env transition to ReplayBuffer

        :env: RL training environment. env.reset() env.step()
        :buffer: Experience Replay Buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explored target_step number of step in env
        :float reward_scale: scale reward, 'reward * reward_scale'
        :float gamma: discount factor, 'mask = 0.0 if done else gamma'
        :return int target_step: collected target_step number of step in env
        """
        for _ in range(target_step):
            action = self.select_action(self.state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            buffer.append_buffer(self.state, other)
            self.state = env.reset() if done else next_s
        return target_step

    def update_net(self, buffer, target_step, batch_size, repeat_times) -> (float, float):
        """update the neural network by sampling batch data from ReplayBuffer

        replace by different DRL algorithms.
        return the objective value as training information to help fine-tuning

        `buffer` Experience replay buffer. buffer.append_buffer() buffer.extend_buffer()
        :int target_step: explore target_step number of step in env
        `int batch_size` sample batch_size of data for Stochastic Gradient Descent
        :float repeat_times: the times of sample batch = int(target_step * repeat_times) in off-policy
        :return float obj_a: the objective value of actor
        :return float obj_c: the objective value of critic
        """

    def save_load_model(self, cwd, if_save):
        """save or load model files

        :str cwd: current working directory, we save model file here
        :bool if_save: save model or load model
        """
        act_save_path = '{}/actor.pth'.format(cwd)
        cri_save_path = '{}/critic.pth'.format(cwd)

        def load_torch_file(network, save_path):
            network_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
            network.load_state_dict(network_dict)

        if if_save:
            if self.act is not None:
                torch.save(self.act.state_dict(), act_save_path)
            if self.cri is not None:
                torch.save(self.cri.state_dict(), cri_save_path)
        elif (self.act is not None) and os.path.exists(act_save_path):
            load_torch_file(self.act, act_save_path)
            print("Loaded act:", cwd)
        elif (self.cri is not None) and os.path.exists(cri_save_path):
            load_torch_file(self.cri, cri_save_path)
            print("Loaded cri:", cwd)
        else:
            print("FileNotFound when load_model: {}".format(cwd))

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update a target network via current network

        :nn.Module target_net: target network update via a current network, it is more stable
        :nn.Module current_net: current network update via an optimizer
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))

    def update_record(self, **kwargs):
        """update the self.train_record for recording the metrics in training process
        :**kwargs :named arguments is the metrics name, arguments value is the metrics value.
        both of them will be prined and showed in tensorboard
        """
        self.train_record.update(kwargs)

class AgentPPO(AgentBase):
    def __init__(self, args=None):
        super().__init__(args)
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
        self.ratio_clip = 0.3 if args is None else args['ratio_clip']
        # could be 0.01 ~ 0.05
        self.lambda_entropy = 0.05 if args is None else args['lambda_entropy']
        # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.lambda_gae_adv = 0.97 if args is None else args['lambda_gae_adv']
        # if use Generalized Advantage Estimation
        self.if_use_gae = True if args is None else args['if_use_gae']
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        self.if_use_dn = False if args is None else args['if_use_dn']
        self.gamma_att = 0.9 if args is None else args['gamma_att']

        self.noise = None
        self.optimizer = None
        self.compute_reward = None  # attribution

    def init(self, InitDict, reward_dim=1, if_per=False, 
                if_load_model=False, actor_path=None, critic_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv
        self.batch_size = InitDict['batch_size']
        if InitDict['use_attbias']:
            self.att_bias = torch.ones([InitDict[ 'block_size_state'],InitDict[ 'block_size'],InitDict['state_dim']])
            for i in range(InitDict[ 'block_size_state']):
                for j in range(InitDict[ 'block_size']):
                    if j == 0:
                        self.att_bias[i,j,:] = self.gamma_att
                    else:
                        self.att_bias[i,j,:] = self.att_bias[i,j-1,:] *  self.gamma_att
            #Joint Positions
            # for i in range(1,9+1):
            #     self.att_bias[:,:,i] = 1
            # #End-effector Positions
            # for i in range(10,18+1):
            #     self.att_bias[:,:,i] = 1
            
            # for i in range(23,35+1):
            #     self.att_bias[:,:,i] = 1
        else:
            self.att_bias = None
        self.cri = CriticAdvAtt(InitDict).to(self.device)
        if if_load_model:
            state_dict = torch.load(critic_path,map_location=self.device)  # 加载模型
            new_state_dict = {}
            for n, p in state_dict.items():
                if n[:13] == 'net_objective':
                    new_state_dict[n] = self.cri.state_dict()[n]
                else:
                    new_state_dict[n] = p
            self.cri.load_state_dict(new_state_dict)
        self.act = ActorPPOAtt(InitDict).to(self.device)
        if if_load_model:
            state_dict = torch.load(actor_path,map_location=self.device)  # 加载模型
            new_state_dict = {}
            for n, p in state_dict.items():
                if n[:13] == 'net_objective':
                    new_state_dict[n] = self.act.state_dict()[n]
                else:
                    new_state_dict[n] = p
            self.act.load_state_dict(new_state_dict)

        # Count variables
        var_counts = tuple(count_vars(module) for module in [self.act])
        logging.info('\nNumber of action net parameters: \t pi: %d'%var_counts)
        var_counts = tuple(count_vars(module) for module in [self.cri])
        logging.info('\nNumber of critic net parameters: \t v: %d\n'%var_counts)

        self.optimizer = torch.optim.AdamW([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER

    @staticmethod
    def select_action(state,state_past, policy, att_bias=None,device='cuda:0'):
        """select action for PPO

       :array state: state.shape==(state_dim, )

       :return array action: state.shape==(action_dim, )
       :return array noise: noise.shape==(action_dim, ), the noise
       """
        states = torch.as_tensor((state,), dtype=torch.float32).detach_().to(device)
        states_past = torch.as_tensor((state_past,), dtype=torch.float32).detach_().to(device)
        if att_bias is not None:
            att_bias = att_bias.to(device)
        if states.ndim == 1:
            states.unsqueeze(0)
        states = states.reshape(states.shape[0],1,-1)
        action = policy.get_action(states,states_past,att_bias)[0]
        return action.detach().cpu().numpy()

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state,buf_state_past = buffer.sample_all()
            buf_state = buf_state.unsqueeze(1)

            bs =  self.batch_size  # set a smaller 'bs: batch size' when out of GPU memory.
            # buf_value = torch.cat([self.cri(buf_state[i:i + bs],buf_state_past[i:i + bs],self.att_bias) for i in range(0, buf_state.size(0), bs)], dim=0)
            # buf_logprob = self.act.compute_logprob(buf_state,buf_state_past,self.att_bias, buf_action).unsqueeze(dim=1)

            buf_value = torch.cat([self.cri(buf_state[i:i + bs],buf_state_past[i:i + bs],self.att_bias) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = self.act.compute_logprob(buf_state,buf_state_past,self.att_bias, buf_action).unsqueeze(dim=1)
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for idx in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            state_past = buf_state_past[indices]
            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            logprob = buf_logprob[indices]
            adv = buf_adv[indices]
            buf_traj = buf_value[indices]

            #new_logprob = self.act.compute_logprob(state,state_past,self.att_bias, action).unsqueeze(dim=1)  # it is obj_actor
            new_logprob = self.act.compute_logprob(state,state_past,self.att_bias, action).unsqueeze(dim=1)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            approx_kl = (logprob - new_logprob).mean().item()

            if approx_kl > 0.01:
                logging.info('Early stopping at step %d due to reaching max kl.'%idx)
                break
            obj_surrogate1 = adv * ratio
            obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

           # value = self.cri(state,state_past,self.att_bias)  # critic network predicts the reward_sum (Q value) of state
            value = self.cri(state,state_past,self.att_bias)  # critic network predicts the reward_sum (Q value) of state
            # clipv的技巧，防止更新前后V差距过大，对其进行惩罚
            value_clip = buf_traj + torch.clamp(value - buf_traj, -self.ratio_clip, self.ratio_clip)
            obj_critic = torch.max(self.criterion(value, r_ret),self.criterion(value_clip, r_ret))
            

            obj_united = obj_actor + obj_critic / (r_ret.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()
        if idx>0:
            self.update_record(obj_a=obj_surrogate.item(),
                            obj_c=obj_critic.item(),
                            obj_tot=obj_united.item(),
                            kl=approx_kl,
                            a_std=self.act.a_std_log.exp().mean().item(),
                            entropy=(-obj_entropy.item()))
        return self.train_record

    def compute_reward_adv(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]
        buf_adv = buf_r_ret - (buf_mask * buf_value)
        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return

        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        pre_adv = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                              device=self.device)  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]

            buf_adv[i] = buf_reward[i] + buf_mask[i] * pre_adv - buf_value[i]
            pre_adv = buf_value[i] + buf_adv[i] * self.lambda_gae_adv

        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv



'''Utils'''


def bt(m):
    return m.transpose(dim0=-2, dim1=-1)


def btr(m):
    return m.diagonal(dim1=-2, dim2=-1).sum(-1)


def gaussian_kl(mu_i, mu, A_i, A):
    """
    decoupled KL between two multivariate gaussian distribution
    C_μ = KL(f(x|μi,Σi)||f(x|μ,Σi))
    C_Σ = KL(f(x|μi,Σi)||f(x|μi,Σ))
    :param μi: (B, n)
    :param μ: (B, n)
    :param Ai: (B, n, n)
    :param A: (B, n, n)
    :return: C_μ, C_Σ: mean and covariance terms of the KL
    """
    n = A.size(-1)
    mu_i = mu_i.unsqueeze(-1)  # (B, n, 1)
    mu = mu.unsqueeze(-1)  # (B, n, 1)
    sigma_i = A_i @ bt(A_i)  # (B, n, n)
    sigma = A @ bt(A)  # (B, n, n)
    sigma_i_inv = sigma_i.inverse()  # (B, n, n)
    sigma_inv = sigma.inverse()  # (B, n, n)
    inner_mu = ((mu - mu_i).transpose(-2, -1) @ sigma_i_inv @ (mu - mu_i)).squeeze()  # (B,)
    inner_sigma = torch.log(sigma_inv.det() / sigma_i_inv.det()) - n + btr(sigma_i_inv @ sigma_inv)  # (B,)
    C_mu = 0.5 * torch.mean(inner_mu)
    C_sigma = 0.5 * torch.mean(inner_sigma)
    return C_mu, C_sigma


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """output a OU-noise

        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])