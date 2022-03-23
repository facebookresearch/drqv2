from pathlib import Path
from replay_buffer import ReplayBufferStorage, make_replay_loader, AbstractReplayBuffer


class DataloaderReplayBuffer(AbstractReplayBuffer):
    def __init__(self, buffer_size, batch_size, nstep, discount,
                 save_snapshot, num_workers, data_specs=None):
        assert data_specs is not None
        self.work_dir = Path.cwd()
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', buffer_size, batch_size, num_workers,
            save_snapshot, nstep, discount)

        self._replay_iter = None

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def add(self, time_step):
        self.replay_storage.add(time_step)

    def __next__(self,):
        return next(self.replay_iter)

    def __len__(self,):
        return len(self.replay_storage)
