import gzip
import pickle
import numpy as np


class BaseSampler:
    def __init__(self, episode, tmp_dir, out_queue):
        self.episode = episode
        self.tmp_dir = tmp_dir
        self.out_queue = out_queue
        self.action_count = [0, 0]
        self.sample_count = 0

    def write_sample(self, state0, state1, action):
        filename = self.tmp_dir + f'/sample_{self.episode}_{self.sample_count}.pkl'
        with gzip.open(filename, 'wb') as f:
            f.write(pickle.dumps({
                'state0': state0,
                'state1': state1,
                'action': action,
            }))

        self.out_queue.put({
            'type': "sample",
            'episode': self.episode,
            'filename': filename,
        }, False)

        self.action_count[action] += 1
        self.sample_count += 1

    def create_sample(self, state0, state1, action):
        # If the statistics functionality is removed
        # from write_sample(), move it here instead
        self.write_sample(state0, state1, action)


class RandomSampler(BaseSampler):
    def __init__(self, episode, tmp_dir, out_queue):
        super().__init__(episode, tmp_dir, out_queue)
        self.random = np.random.default_rng(0)

    def create_sample(self, state0, state1, action):
        if self.random.random() < 0.5:
            self.write_sample(state0, state1, action)
        else:
            self.write_sample(state1, state0, 1 - action)


class DoubleSampler(BaseSampler):
    def __init__(self, episode, tmp_dir, out_queue):
        super().__init__(episode, tmp_dir, out_queue)

    def create_sample(self, state0, state1, action):
        self.write_sample(state0, state1, action)
        self.write_sample(state1, state0, 1 - action)
