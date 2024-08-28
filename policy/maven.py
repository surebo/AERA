import torch
import torch.nn as nn
import torch.nn.functional as f
import os
from network.maven_net import HierarchicalPolicy, BootstrappedRNN, VarDistribution
from network.qmix_net import QMixNet


class MAVEN:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape

        # input shaoe of rnn
        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # network
        self.z_policy = HierarchicalPolicy(args)  # choose z

        self.eval_rnn = BootstrappedRNN(input_shape, args)  # choose action for each agent
        self.target_rnn = BootstrappedRNN(input_shape, args)

        self.eval_qmix_net = QMixNet(args)  # mix the q value
        self.target_qmix_net = QMixNet(args)

        self.mi_net = VarDistribution(args)  # get q(z|sigma(tau))

        self.args = args
        if self.args.cuda:
            self.z_policy.cuda()
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
            self.mi_net.cuda()
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_z_policy = self.model_dir + '/z_policy_params.pkl'
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qmix = self.model_dir + '/qmix_net_params.pkl'
                path_mi = self.model_dir + '/mi_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.z_policy.load_state_dict(torch.load(path_z_policy, map_location=map_location))
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                self.mi_net.load_state_dict(torch.load(path_mi, map_location=map_location))
                print('Successfully load the model: {}, {}, {} and {}'.format(path_z_policy, path_rnn, path_qmix, path_mi))
            else:
                raise Exception("No model!")

      
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.z_policy.parameters()) + list(self.eval_qmix_net.parameters()) +\
                               list(self.eval_rnn.parameters()) + list(self.mi_net.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg MAVEN')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated, z = batch['s'], batch['s_next'], batch['u'], batch['r'],  \
                                                                batch['avail_u'], batch['avail_u_next'],\
                                                                batch['terminated'],   batch['z']
        mask = 1 - batch["padded"].float() 
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            avail_u = avail_u.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
            z = z.cuda()
        # -------------------------------------------------RL Loss------------------------------------------------------
        z_prob = self.z_policy(s[:, 0, :])
        log_z_prob = torch.log(z_prob)
        entropy = -(z_prob * log_z_prob).sum(dim=-1).detach()
        class_z = z.long().argmax(dim=-1, keepdim=True)  # transform onehot vector to int
        z_prob_taken = torch.gather(z_prob, index=class_z, dim=-1).squeeze(-1)
        z_returns = r.sum(dim=1)
        rl_loss = - (z_prob_taken * (z_returns + self.args.entropy_coefficient * entropy)).mean()

        # -------------------------------------------------RL Loss------------------------------------------------------

        # -------------------------------------------------MI Loss------------------------------------------------------
        inputs = []
        for i in range(episode_num):
            length = int(mask[i].sum().item())
            q, avail_action = q_evals[i, :length], avail_u[i, :length]

            # only take the valid data
            q = f.softmax(q, dim=-1)
            q = q * avail_action
            q = q / q.sum(dim=-1, keepdim=True)
            q = torch.cat([q, torch.zeros_like(q_evals[i, length:])], dim=0)
            q = q.reshape(max_episode_len, -1)
            inputs.append(q)
        inputs = torch.stack(inputs, dim=0)
        inputs = torch.cat([inputs, s], dim=-1)
        inputs = inputs.permute(1, 0, 2)

        mi_prob = self.mi_net(inputs)  # (episode_num, args.noise_dim)
        class_z = class_z.squeeze(-1)
        mi_loss = f.cross_entropy(mi_prob, class_z)
        # -------------------------------------------------MI Loss------------------------------------------------------

        # -------------------------------------------------QL Loss------------------------------------------------------
 
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)


        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        td_error = (q_total_eval - targets.detach())
        masked_td_error = mask * td_error  
        ql_loss = (masked_td_error ** 2).sum() / mask.sum()
        # -------------------------------------------------QL Loss------------------------------------------------------

        # loss = rl_loss + self.args.lambda_mi * mi_loss + self.args.lambda_ql * ql_loss
        loss = self.args.lambda_mi * mi_loss + self.args.lambda_ql * ql_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        # print('mi_loss is {}, ql_loss is {}'.format(mi_loss, ql_loss))
        # print('Training params:')
        # for params in self.eval_rnn.named_parameters():
        #     print(params)

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
  

        if self.args.last_action:
            if transition_idx == 0:  
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        z = batch['z'].repeat(self.n_agents, 1)  # each episode has one z
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # add last_action、agent_id
            if self.args.cuda:
                z = z.cuda()
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden, z)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden, z)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.z_policy.state_dict(), self.model_dir + '/' + num + '_z_policy_params.pkl')
        torch.save(self.mi_net.state_dict(),  self.model_dir + '/' + num + '_mi_net_params.pkl')
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_qmix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')
