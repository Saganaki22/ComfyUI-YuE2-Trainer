"""Display-only registration of the training model for memory visualizers."""
import sys
import types

import pytest

sys.path.insert(0, r'C:\Users\drbaph\Documents\ComfyUI')
torch = pytest.importorskip('torch')
mm = pytest.importorskip('comfy.model_management')

from mothersuperior_nodes import _register_training_model, _unregister_training_model


class _FakeLM:
    """Minimal stand-in for the built AR model (parameters + model.model)."""

    def __init__(self):
        self.inner = torch.nn.Sequential(torch.nn.Linear(8, 8))
        self.weight = torch.nn.Parameter(torch.zeros(8, 8))

    def parameters(self):
        return [self.weight]

    def model(self):
        return self.inner

    # model.model attribute access
    class _M:
        pass


def _make_fake():
    lm = types.SimpleNamespace()
    lm.inner = torch.nn.Sequential(torch.nn.Linear(8, 8)).cuda()
    lm.weight = torch.nn.Parameter(torch.zeros(8, 8, device='cuda'))
    lm.parameters = lambda: [lm.weight]
    lm.model = lm.inner
    return lm


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_registration_shows_in_loaded_models():
    fake = _make_fake()
    before = len(mm.current_loaded_models)
    patcher = _register_training_model(fake, rank=4)
    try:
        assert patcher is not None
        assert any(getattr(lm.model, 'model', None) is fake.model or lm.model is patcher
                   for lm in mm.current_loaded_models)
        assert patcher.model_size() > 0
        assert not patcher.is_dynamic()
    finally:
        _unregister_training_model(patcher)
    assert len(mm.current_loaded_models) <= before


def test_registration_returns_none_on_failure():
    # No CUDA model at all -> parameters() raising must be swallowed.
    bad = types.SimpleNamespace(parameters=lambda: (_ for _ in ()).throw(RuntimeError('no gpu')),
                                model=None)
    assert _register_training_model(bad, rank=4) is None
    # Unregister of None is a no-op.
    _unregister_training_model(None)
