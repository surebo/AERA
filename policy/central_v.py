import torch
import os
from network.base_net import RNN, Critic
from network.commnet import CommNet
from network.g2anet import G2ANet


class CentralV:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape  # actor
        critic_input_shape = self.state_shape  # critic
    
        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        self.args = args

        if self.args.alg == 'central_v':
            self.eval_rnn = RNN(actor_input_shape, args)
            print('Init alg central_v')
        elif self.args.alg == 'central_v+commnet':
            self.eval_rnn = CommNet(actor_input_shape, args)
            print('Init alg central_v+commnet')
        elif self.args.alg == 'central_v+g2anet':
            print('Init alg central_v+g2anet')
            self.eval_rnn = G2ANet(actor_input_shape, args)
        else:
            raise Exception("No such algorithm")

        self.eval_critic = Critic(critic_input_shape, self.args)
        self.target_critic = Critic(critic_input_shape, self.args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_params.pkl'):
                path_rnn = self.model_dir + '/rnn_params.pkl'
                path_critic = self.model_dir + '/critic_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_critic.load_state_dict(torch.load(path_critic, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_critic))
            else:
                raise Exception("No model!")

        self.target_critic.load_state_dict(self.eval_critic.state_dict())

        self.rnn_parameters = list(self.eval_rnn.parameters())
        self.critic_parameters = list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.critic_optimizer = torch.optim.RMSprop(self.critic_parameters, lr=args.lr_critic)
            self.rnn_optimizer = torch.optim.RMSprop(self.rnn_parameters, lr=args.lr_actor)
        self.args = args


        self.eval_hidden = None

    def learn(self, batch, max_episode_len, train_step, epsilon):  
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, terminated = batch['u'], batch['r'],  batch['avail_u'], batch['terminated']
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)  
        if self.args.cuda:
            u = u.cuda()
            mask = mask.cuda()

        td_error = self._train_critic(batch, max_episode_len, train_step)
        td_error = td_error.repeat(1, 1, self.n_agents)

        action_prob = self._get_action_prob(batch, max_episode_len, epsilon)

        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        pi_taken[mask == 0] = 1.0  
        log_pi_taken = torch.log(pi_taken)


        loss = - ((td_error.detach() * log_pi_taken) * mask).sum() / mask.sum()
        self.rnn_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnn_parameters, self.args.grad_norm_clip)
        self.rnn_optimizer.step()
        # print('Actor loss is', loss)

    def _get_v_values(self, batch, max_episode_len):
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = batch['s'][:, transition_idx], batch['s_next'][:, transition_idx],
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            v_eval = self.eval_critic(inputs)
            v_target = self.target_critic(inputs_next)

            v_evals.append(v_eval)
            v_targets.append(v_target)
        v_evals = torch.stack(v_evals, dim=1)  # (episode_num, max_episode_len, 1)
        v_targets = torch.stack(v_targets, dim=1)
        return v_evals, v_targets

    def _get_actor_inputs(self, batch, transition_idx):
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)
        if self.args.last_action:
            if transition_idx == 0:  
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']  
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)  
            if self.args.cuda:
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            outputs, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)
        action_prob = torch.stack(action_prob, dim=1).cpu()

        action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])  
        action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        action_prob[avail_actions == 0] = 0.0
        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def _train_critic(self, batch, max_episode_len, train_step):
        r, terminated = batch['r'], batch['terminated']
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents) 
        if self.args.cuda:
            mask = mask.cuda()
            r = r.cuda()
            terminated = terminated.cuda()
        v_evals, v_next_target = self._get_v_values(batch, max_episode_len)

        targets = r + self.args.gamma * v_next_target * (1 - terminated)
        td_error = targets.detach() - v_evals
        masked_td_error = mask * td_error  
        loss = (masked_td_error ** 2).sum() / mask.sum()
        # print('Critic Loss is ', loss)
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
        return td_error

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')