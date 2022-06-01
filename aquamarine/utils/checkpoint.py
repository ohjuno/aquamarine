from typing import Optional

import os.path
import torch


class CKPT:

    __base_folder__ = 'checkpoint'

    def __init__(self, path: str, ckpt,
                 save_best_only: bool = True, last_epoch: int = -1):
        self.path = os.path.join(path, self.__base_folder__)
        self.ckpt = ckpt
        self.last_epoch = last_epoch

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def __call__(self, tag: str, epoch: Optional[int] = None):
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        filename = os.path.join(self.path, tag)
        torch.save(self.ckpt, filename)


if __name__ == '__main__':

    checkpoint = CKPT('/tmp/pycharm_project_aquamarine', {'epoch': None})
    checkpoint('test.pt')

    breakpoint()
