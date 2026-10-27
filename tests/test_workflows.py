"""File workflow GitHub Actions phải hợp lệ: GitHub từ chối khoá trùng / thụt lề sai, còn PyYAML
mặc định lặng lẽ chấp nhận khoá trùng — nên kiểm tra nghiêm ngặt ở đây trước khi push."""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).parents[1] / ".github" / "workflows").glob("*.yml"))


class StrictLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    seen = set()
    for k, _ in node.value:
        key = loader.construct_object(k, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(None, None, f"khoá trùng {key!r}", k.start_mark)
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_is_valid(path):
    d = yaml.load(path.read_text(encoding="utf-8"), Loader=StrictLoader)
    on = d.get("on", d.get(True))
    assert on, "thiếu 'on:'"
    if isinstance(on, dict) and isinstance(on.get("push"), dict):
        assert set(on["push"]) <= {"branches", "tags", "paths", "branches-ignore", "paths-ignore", "tags-ignore"}
    for job in d["jobs"].values():
        assert "runs-on" in job
        for step in job["steps"]:
            assert ("run" in step) != ("uses" in step), f"bước sai: {step.get('name')}"
