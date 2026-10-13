"""Fast unit tests that need no weights or slides."""
import numpy as np
import pytest

from himico.cohort import Cohort, slide_id
from himico.config import PipelineConfig, template_yaml
from himico.wsi import mask_fraction, tile_grid


def test_slide_id_strips_extensions():
    assert slide_id("/a/CRC05-registered.ome.tiff") == "CRC05-registered"
    assert slide_id("TCGA-AA-1234-01Z-00-DX1.ABC.svs") == "TCGA-AA-1234-01Z-00-DX1.ABC"


def test_cohort_patients_and_shards(tmp_path):
    for n in ["TCGA-AA-0001-01Z-DX1.svs", "TCGA-AA-0001-01Z-DX2.svs", "TCGA-BB-0002-01Z-DX1.svs"]:
        (tmp_path / n).write_bytes(b"x")
    c = Cohort.from_folder(tmp_path, patient=r"^(TCGA-\w\w-\w{4})")
    assert sorted(c.patients) == ["TCGA-AA-0001", "TCGA-BB-0002"]
    parts = [set(c.shard(i, 2).slide_ids) for i in range(2)]
    assert parts[0] | parts[1] == set(c.slide_ids) and not parts[0] & parts[1]
    assert all(len({s.patient for s in c.shard(i, 2)}) == 1 for i in range(2))   # patients never split


def test_mask_fraction():
    mask = np.zeros((100, 100), np.uint8)
    mask[:, :50] = 255                                  # left half is tissue
    grid = tile_grid(1000, 1000, 250)                   # 4x4 tiles over a 1000 px image
    f = mask_fraction(mask, grid, (1000, 1000)).reshape(4, 4)
    assert np.allclose(f[:, :2], 1) and np.allclose(f[:, 2:], 0)
    edge = mask_fraction(np.full((100, 100), 255, np.uint8), np.array([[900, 0, 1100, 200]]), (1000, 1000))
    assert edge[0] == pytest.approx(0.5)                # half of the tile hangs over the slide edge


def test_config_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(template_yaml())
    c = PipelineConfig.from_yaml(p)
    assert c.workspace.as_posix() == "himico_runs/my_cohort/run_1" and c.mask.keep_classes == [1, 2, 5, 6]
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"not_a_key": 1})


def test_api_key_storage_and_precedence(tmp_path, monkeypatch):
    from himico import auth
    monkeypatch.setenv("HIMICO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("HIMICO_API_KEY", raising=False)
    monkeypatch.delenv("INSIGHT_SERVER_TOKEN", raising=False)
    assert auth.api_key("http://a") is None
    p = auth.save_key("k1", "http://a/")
    assert p.stat().st_mode & 0o077 == 0 and auth.api_key("http://a") == "k1" and auth.api_key("http://b") is None
    monkeypatch.setenv("HIMICO_API_KEY", "env")
    assert auth.api_key("http://a") == "env"
    monkeypatch.delenv("HIMICO_API_KEY")
    assert auth.forget_key("http://a") and auth.api_key("http://a") is None


def test_backend_resolution(monkeypatch):
    from himico.models.remote import resolve_backend
    monkeypatch.setenv("HIMICO_SERVER", "https://example.org/")
    assert resolve_backend("local") is None
    assert resolve_backend("himico") == "https://example.org"
    assert resolve_backend("http://gpu:8080/") == "http://gpu:8080"
    with pytest.raises(ValueError):
        resolve_backend("gpu-box")


def test_server_keys_store_only_hashes(tmp_path):
    from himico.server import _hash, add_key, load_keys, revoke_key
    f = tmp_path / "keys.json"
    key = add_key(f, "Jane", "academic", ["insight-crc"])
    keys = load_keys(f)
    assert key not in f.read_text() and keys[_hash(key)]["tier"] == "academic"
    assert revoke_key(f, keys[_hash(key)]["id"]) and load_keys(f) == {}
