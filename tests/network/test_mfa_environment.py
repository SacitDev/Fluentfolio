"""Build the optional aligner's real locked environment, without downloading speech models."""

from pathlib import Path

import pytest

from fluentfolio.alignment import mfa
from fluentfolio.models import catalogue, download
from fluentfolio.models.catalogue import SYSTEM
from fluentfolio.setup import mfa_env


@pytest.mark.network
def test_the_patched_mfa_environment_builds_and_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = tmp_path / "app"
    monkeypatch.setenv("FLUENTFOLIO_APP_ROOT", str(app))
    component = catalogue.get("mfa")
    tool = next(f for f in component.files_for(SYSTEM) if f.path.startswith("micromamba"))
    target = mfa_env.micromamba(app)
    download.fetch_file(component, tool, target)
    if SYSTEM == "linux":
        target.chmod(0o755)
    assert mfa_env.create(app) == "3.4.3"
    assert mfa.env_current(app)
    assert mfa_env.installed_version(app) == "3.4.3"
