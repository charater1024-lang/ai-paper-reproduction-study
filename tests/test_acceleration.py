from __future__ import annotations

import pytest
import torch
from torch import nn

from llm_engineering_lab.acceleration import get_accelerator, resolve_device


def test_cpu_override_moves_nested_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_LAB_DEVICE", "cpu")
    monkeypatch.setenv("AI_LAB_AMP", "auto")
    monkeypatch.setenv("AI_LAB_CPU_THREADS", "2")

    accelerator = get_accelerator()
    model = nn.Linear(3, 2)
    payload = {"features": torch.ones(4, 3), "ids": ["row-1"]}
    moved_model, moved_payload = accelerator.move(model, payload)

    assert accelerator.device.type == "cpu"
    assert not accelerator.amp_enabled
    assert torch.get_num_threads() == 2
    assert next(moved_model.parameters()).device.type == "cpu"
    assert moved_payload["features"].device.type == "cpu"
    assert moved_payload["ids"] == ["row-1"]


def test_auto_device_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_LAB_DEVICE", raising=False)
    expected = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert resolve_device().type == expected


def test_explicit_unavailable_cuda_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Install_GPU_PyTorch.cmd"):
        resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_backward_optimizer() -> None:
    accelerator = get_accelerator("cuda", amp="0")
    model = nn.Linear(8, 2).to(accelerator.device)
    inputs = torch.randn(16, 8, device=accelerator.device)
    targets = torch.randint(0, 2, (16,), device=accelerator.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    optimizer.step()
    accelerator.synchronize()

    assert torch.isfinite(loss)
    assert not torch.equal(before, model.weight)
