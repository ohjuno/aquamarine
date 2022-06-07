from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import torch

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler


def load_batch(
        batch: Sequence[Tensor],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False
) -> Tuple[Union[Tensor, Sequence, Mapping, str, bytes], ...]:
    return batch if device is None else (b.to(device=device, non_blocking=non_blocking) for b in [*batch])


def update(
        model: Module,
        optimizer: Optimizer,
        criterion: Module,
        load_batch_fn: Callable = load_batch,
        output_format: Callable = lambda i, t, p, l: l.item(),
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
) -> Callable:
    r"""Return factory function that updates the model with single batch

    Args:
        ...
    """
    def update_batch(batch: Sequence[Tensor]) -> Union[Any, Tuple[Tensor]]:
        model.train()
        criterion.train()
        inputs, targets = load_batch_fn(batch, device=device, non_blocking=non_blocking)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        return output_format(inputs, targets, outputs, loss)

    return update_batch


def update_with_amp(
        model: Module,
        optimizer: Optimizer,
        criterion: Module,
        scaler: GradScaler,
        load_batch_fn: Callable = load_batch,
        output_format: Callable = lambda i, t, p, l: l.item(),
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
) -> Callable:
    r"""Return factory function that updates the model using ``torch.cuda.amp`` with single batch

    Args:
        ...
    """
    def update_batch(batch: Sequence[Tensor]) -> Union[Any, Tuple[Tensor]]:
        model.train()
        criterion.train()
        inputs, targets = load_batch_fn(batch, device=device, non_blocking=non_blocking)
        optimizer.zero_grad()
        with autocast(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return output_format(inputs, targets, outputs, loss)

    return update_batch


def evaluate(
        model: Module,
        load_batch_fn: Callable = load_batch,
        output_format: Callable = lambda i, t, p: (p, t),
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
) -> Callable:
    r"""Return factory function that evaluates the performance of the model with single batch

    Args:
        ...
    """
    def evaluate_batch(batch: Sequence[Tensor]) -> Union[Any, Tuple[Tensor]]:
        model.eval()
        with torch.no_grad():
            inputs, targets = load_batch_fn(batch, device=device, non_blocking=non_blocking)
            outputs = model(inputs)
            return output_format(inputs, targets, outputs)

    return evaluate_batch


def evaluate_with_amp(
        model: Module,
        load_batch_fn: Callable = load_batch,
        output_format: Callable = lambda i, t, p: (p, t),
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
) -> Callable:
    r"""Return factory function that evaluates the performance of the model using ``torch.cuda.amp`` with single batch

    Args:
        ...
    """
    def evaluate_batch(batch: Sequence[Tensor]) -> Union[Any, Tuple[Tensor]]:
        model.eval()
        with torch.no_grad():
            inputs, targets = load_batch_fn(batch, device=device, non_blocking=non_blocking)
            with autocast(enabled=True):
                outputs = model(inputs)
            return output_format(inputs, targets, outputs)

    return evaluate_batch
