from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization for Efficiently Improving Generalization.
    https://arxiv.org/abs/2010.01412
    """

    def __init__(
            self,
            params: _params_t,
            base_optimizer: Callable,
            rho: float = 0.05,
            adaptive: bool = False,
            **kwargs
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Rho {rho} must be positive")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']
        self.base_optimizer.step()

    @torch.no_grad()
    def step(
            self,
            closure: Optional[Callable[[], float]] = None
    ):
        if closure is None:
            raise ValueError("Sharpness Aware Minimization requires closure, but it was not provided.")

        closure = torch.enable_grad()(closure)

        self.zero_grad()
        self.first_step()
        self.zero_grad()
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group['param'] if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict) -> None:
        super(SAM, self).load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
