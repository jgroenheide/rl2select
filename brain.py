import model as ml
import numpy as np
import torch as th
import torch.utils.data


class Brain:
    """
    Brain class. Holds the policy, and receives requests from the agents to sample actions using it,
    given a state. It also performs policy updates after receiving training samples from the agents.
    """

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.actor = ml.MLPPolicy().to(device)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=self.config['lr'])
        self.random = np.random.RandomState(seed=self.config['seed'])

    def sample_actions(self, requests):
        receivers = [r['receiver'] for r in requests]
        for r in requests: del r['receiver']

        # process all requests in a batch
        request_loader = th.utils.data.DataLoader(requests, batch_size=self.config['batch_size'])

        actions = []
        for batch in request_loader:
            with th.no_grad():
                # batch = batch.to(self.device)
                output = self.actor(*batch['state'])
                g_actions = th.round(output)
                e_actions = th.rand_like(output) < output
                tmp = batch['greedy'].unsqueeze(dim=1)
                actions += th.where(tmp, g_actions, e_actions)

            # for action, receiver in zip(actions, receivers):
            #     receiver.put(action)

        responses = th.concatenate(actions).tolist()
        for receiver, response in zip(receivers, responses):
            receiver.put(response)

    def update(self, transitions):
        stats = {'loss': 0.0, 'reinforce_loss': 0.0, 'entropy': 0.0}

        n_samples = len(transitions)
        if n_samples < 1:
            return stats

        # transitions = torch_geometric.loader.DataLoader(transitions, batch_size=16, shuffle=True)
        transition_loader = torch.utils.data.DataLoader(transitions, batch_size=16, shuffle=True)

        self.actor_optimizer.zero_grad()
        for batch in transition_loader:
            # batch = batch.to(self.device)
            loss = th.tensor([0.0], device=self.device)
            action_prob = self.actor(*batch['state'])
            dist = th.distributions.bernoulli.Bernoulli(action_prob)

            # REINFORCE / actor loss
            returns = batch['returns'].float()
            actions = batch['action'].float()
            reinforce_loss = - (returns * dist.log_prob(actions)).sum()
            reinforce_loss /= n_samples
            loss += reinforce_loss

            # ENTROPY
            entropy = dist.entropy().sum()
            entropy /= n_samples
            loss += - self.config['entropy_bonus'] * entropy

            loss.backward()

            # Update stats
            stats['loss'] += loss.item()
            stats['reinforce_loss'] += reinforce_loss.item()
            stats['entropy'] += entropy.item()

        self.actor_optimizer.step()

        return stats

    def save(self, filename):
        # Save in the same directory as the pretrained params
        th.save(self.actor.state_dict(), filename)
