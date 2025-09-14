import os
import sys
import types
import pytest

# Ensure repository root in sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import utils


class DummyPlantUML:
    def __init__(self, url=None):
        self.url = url

    def processes(self, *args, **kwargs):
        # Return some fake PNG bytes regardless of signature used
        return b"\x89PNG\r\n\x1a\n\x00dummy"


def test_render_writes_inside_artifacts(monkeypatch):
    # Monkeypatch PlantUML to avoid network calls
    monkeypatch.setattr(utils, "PlantUML", DummyPlantUML)
    out_rel = "diagrams/test_plantuml.png"
    utils.render_plantuml_diagram("@startuml\nA -> B: hi\n@enduml", out_rel)

    expected = os.path.join(utils._find_project_root(), "artifacts", out_rel)
    assert os.path.exists(expected)
    assert os.path.getsize(expected) > 0


def test_render_accepts_artifacts_prefixed_path(monkeypatch):
    monkeypatch.setattr(utils, "PlantUML", DummyPlantUML)
    out_rel = "artifacts/diagrams/test_plantuml_prefixed.png"
    utils.render_plantuml_diagram("@startuml\nA -> B: hi\n@enduml", out_rel)

    expected = os.path.join(utils._find_project_root(), out_rel)
    assert os.path.exists(expected)
    assert os.path.getsize(expected) > 0


@pytest.mark.parametrize("bad", ["../outside.png", "/tmp/out.png"]) 
def test_render_rejects_outside_artifacts(monkeypatch, bad):
    monkeypatch.setattr(utils, "PlantUML", DummyPlantUML)
    with pytest.raises(ValueError):
        utils.render_plantuml_diagram("@startuml\nA->B\n@enduml", bad)

