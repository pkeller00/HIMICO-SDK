"""Model weights: a local folder first, otherwise Hugging Face (with the user's own login for gated repos).

Preprocessing weights (GrandQC, CRC tissue classifier) are public. UNI is gated by its authors: request access and run
`hf auth login` (or set HF_TOKEN). Prediction-model weights (INSIGHT, ...) are private and not downloaded: the
pipeline sends patient graphs to the HIMICO server (see himico.auth) unless you were given the weights and put them
in a local `weights_dir`. No access tokens live in this package.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("himico")

PREPROCESSING_REPO = "pkeller01/INSIGHT-preprocessing"
INSIGHT_REPO = "pkeller01/INSIGHT"
UNI_REPO = "MahmoodLab/UNI"


@dataclass(frozen=True)
class WeightFile:
    key: str            # path inside a local weights_dir, and inside the repo
    repo: str
    repo_path: str
    sha256: str | None = None
    note: str = ""      # licence / access note shown when the file cannot be fetched


_GRANDQC = "GrandQC (Weng et al., Nat Commun 2024), CC BY-NC-SA 4.0"
FILES: dict[str, WeightFile] = {f.key: f for f in [
    WeightFile("grandqc/td_mpp10.safetensors", PREPROCESSING_REPO, "grandqc/td_mpp10.safetensors",
               "2b1ff2227de7743ec17d2f8b8991d3c90b1515e19b43ff88f79901c24ee23897", _GRANDQC),
    WeightFile("grandqc/qc_mpp1.safetensors", PREPROCESSING_REPO, "grandqc/qc_mpp1.safetensors",
               "6d66d04391c3228201dc220fa3cecde0aa46a69cf9fcbdc22c92bcd36b7c60aa", _GRANDQC),
    WeightFile("grandqc/qc_mpp1.5.safetensors", PREPROCESSING_REPO, "grandqc/qc_mpp1.5.safetensors",
               "28c3d10b1e530809cc4ea76b66d5b35bc3552e250513423aad2a04528a670c0e", _GRANDQC),
    WeightFile("grandqc/qc_mpp2.safetensors", PREPROCESSING_REPO, "grandqc/qc_mpp2.safetensors",
               "b0fac1fdb6aa49dade415e41a6a6beaa19005b6dfac31517a34428f515cdc521", _GRANDQC),
    WeightFile("crc_tissue/vit_b16.safetensors", PREPROCESSING_REPO, "crc_tissue/vit_b16.safetensors",
               "c13051b569e40ac9e83628630cc001998c0e5004b74052cdf7640bde378b42c6",
               "INSIGHT CRC tissue classifier, CC BY-NC-ND 4.0"),
    WeightFile("verify/TCGA-4N-A93T.g", PREPROCESSING_REPO, "verify/TCGA-4N-A93T.g",
               "445b375470535d38f83116a74f6ca2db17fad79e8e351c5294a122c71418d951", "public TCGA example graph for `himico verify`"),
    WeightFile("uni/pytorch_model.bin", UNI_REPO, "pytorch_model.bin", None,
               "UNI is gated by its authors: request access at https://huggingface.co/MahmoodLab/UNI"),
    *[WeightFile(f"insight/fold{k}/{name}", INSIGHT_REPO, f"fold{k}/{name}", sha,
                 "INSIGHT weights are private: predictions run on the HIMICO server (`himico login`), or put weights "
                 "you were given in --weights-dir")
      for k, sha in zip(range(1, 5), ["faed11049dfdc0927ea70749004d4b78bce83cce7832da463e7f178c09bdfbf8",
                                      "ce4e40bf3538c3fa79c6ac54e4a335b93e1440dd9afd52110968eb4c68564125",
                                      "e10c43e3aa5a2334c51130fa5385a3fb87a8adc47fdb66eb06650665e3855bc9",
                                      "ec1795c1524f0e23681ce5389bfe54169cd0ed570f346a2b0b4ccb04cfced86f"])
      for name, sha in [("model.safetensors", sha), ("minmax.csv", None)]],
]}


class WeightsUnavailable(RuntimeError):
    pass


def private(key: str) -> bool:
    """Weights that are never downloaded for users (only the HIMICO server and named collaborators hold them)."""
    return FILES[key].repo == INSIGHT_REPO


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(key: str, weights_dir: str | Path | None = None, verify: bool = True) -> Path:
    """Local path to a weight file, downloading it from Hugging Face if it is not in `weights_dir`."""
    spec = FILES[key]
    if weights_dir:
        local = Path(weights_dir) / key
        if local.exists():
            return local
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError as e:  # pragma: no cover
        raise WeightsUnavailable("huggingface_hub is not installed") from e
    try:
        path = Path(hf_hub_download(spec.repo, spec.repo_path))
    except GatedRepoError as e:
        raise WeightsUnavailable(f"{key}: access not granted. {spec.note}" + ("" if private(key) else ". Then run `hf auth login`.")) from e
    except RepositoryNotFoundError as e:
        raise WeightsUnavailable(
            f"{key}: repository {spec.repo} not found or not accessible. {spec.note}. "
            f"If you have the file locally, put it at <weights_dir>/{key}.") from e
    if verify and spec.sha256 and _sha256(path) != spec.sha256:
        raise WeightsUnavailable(f"{key}: checksum mismatch for {path}")
    return path


def available(keys, weights_dir: str | Path | None = None) -> bool:
    """True if every file is in `weights_dir` or the Hugging Face cache (no network access)."""
    from huggingface_hub import try_to_load_from_cache
    return all((weights_dir and (Path(weights_dir) / k).exists())
               or isinstance(try_to_load_from_cache(FILES[k].repo, FILES[k].repo_path), str) for k in keys)


def status(weights_dir: str | Path | None = None) -> list[tuple[str, str]]:
    """(key, 'local' | 'cached' | 'missing') for every known weight file, without downloading anything."""
    from huggingface_hub import try_to_load_from_cache
    out = []
    for key, spec in FILES.items():
        if weights_dir and (Path(weights_dir) / key).exists():
            out.append((key, "local"))
            continue
        cached = try_to_load_from_cache(spec.repo, spec.repo_path)
        out.append((key, "cached" if isinstance(cached, str) else "missing"))
    return out
