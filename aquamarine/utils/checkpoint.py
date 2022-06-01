from typing import Dict, List, Optional, Union

import copy
import os.path
import numpy as np
import torch

from torch import Tensor


class CKPT:

    base_folder = 'ckpt'

    def __init__(
            self,
            dirs: Union[str, os.PathLike],
            ckpt: Dict,
            save_best_only: bool = True,
            save_weights_only: bool = True,
            last_epoch: int = -1,
    ) -> None:
        self.path = os.path.join(dirs, self.base_folder)
        self.ckpt = ckpt
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.last_epoch = last_epoch

        # for `save_best_only` option
        self.best_score = torch.inf
        self.best_default_name = 'checkpoint_best_score.pth'

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def _save_checkpoint(self, ckpt, name) -> None:
        torch.save(ckpt, os.path.join(self.path, name))

    def _check_and_update_best_score(self, score: float) -> bool:
        is_best = self.best_score > score
        self.best_score = min(self.best_score, score)
        return is_best

    def step(self, score: Optional[Union[float, List]] = None, epoch: Optional[int] = None) -> None:
        epoch = self.last_epoch + 1 if epoch is None else epoch
        score = np.nanmean(score) if isinstance(score, list) else score

        # make snapshot of checkpoint
        snapshot = dict(epoch=epoch, score=score)
        if self.save_weights_only:
            for k, v in self.ckpt.items():
                snapshot.update({k: v.state_dict()}) if hasattr(v, 'state_dict') else snapshot.update({k: v})
        else:
            ckpt = copy.deepcopy(self.ckpt)
            snapshot.update(ckpt)

        # save checkpoint by the policy
        if self.save_best_only and score is not None:
            is_best = self._check_and_update_best_score(score)
            if is_best:
                self._save_checkpoint(snapshot, self.best_default_name)
        elif not self.save_best_only and score is not None:
            self._save_checkpoint(snapshot, f'checkpoint_epoch_{epoch:04d}_score_{score:.3f}.pth')
        else:
            self._save_checkpoint(snapshot, f'checkpoint_epoch_{epoch:04d}.pth')

        self.last_epoch = np.floor(epoch)
