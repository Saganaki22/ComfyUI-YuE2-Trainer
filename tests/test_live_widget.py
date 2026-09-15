"""Live-widget plumbing: report callback, node reporter, schema outputs."""
import inspect
import sys
import types

import pytest


def test_train_accepts_report():
    from trainer_core.mothersuperior import ar_train
    params = inspect.signature(ar_train.train).parameters
    assert 'report' in params
    assert params['report'].default is None


def test_make_report_none_without_server():
    from mothersuperior_nodes import make_report
    assert make_report(None) is None


def test_make_report_sends_events(monkeypatch):
    sent = []

    class FakeServer:
        instance = None

        def send_sync(self, event, data):
            sent.append((event, data))

    fake = FakeServer()
    fake.instance = fake
    monkeypatch.setitem(sys.modules, 'server', types.SimpleNamespace(PromptServer=fake))

    from mothersuperior_nodes import make_report
    report = make_report('node-42')
    assert report is not None
    report({'type': 'point', 'step': 3, 'artist': 5.5})
    assert sent == [('yue2.training.progress', {'node': 'node-42', 'type': 'point', 'step': 3, 'artist': 5.5})]

    # A failing send must never break training.
    def boom(event, data):
        raise RuntimeError('websocket gone')

    fake.send_sync = boom
    report({'type': 'eval', 'step': 4})
    assert len(sent) == 1


def test_trainer_node_schema_outputs(monkeypatch):
    monkeypatch.setitem(sys.modules, 'folder_paths',
                        types.SimpleNamespace(get_filename_list=lambda kind: []))
    from mothersuperior_nodes import YuE2ArtistARLoRATrainer
    spec = YuE2ArtistARLoRATrainer.INPUT_TYPES()
    assert 'unique_id' in spec.get('hidden', {})
    assert YuE2ArtistARLoRATrainer.RETURN_TYPES == ('STRING', 'STRING', 'STRING')
    assert YuE2ArtistARLoRATrainer.RETURN_NAMES == ('ar_lora_path', 'training_log', 'training_log_path')
