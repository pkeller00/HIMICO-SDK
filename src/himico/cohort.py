"""Cohorts: which slides to process and which patient each belongs to."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

SLIDE_EXTENSIONS = (
    ".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif", ".svslide", ".tif", ".tiff", ".dcm", ".qptiff", ".czi",
)


def slide_id(path: str | Path) -> str:
    """File name without slide extensions, e.g. CRC05.ome.tiff -> CRC05."""
    name = Path(path).name
    for ext in sorted(SLIDE_EXTENSIONS, key=len, reverse=True):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return re.sub(r"\.ome$", "", name, flags=re.I)


@dataclass(frozen=True)
class Slide:
    id: str
    path: Path
    patient: str


class Cohort:
    """An ordered set of slides with a patient for each.

    Build one from a folder (`Cohort.from_folder`) or a CSV manifest (`Cohort.from_csv`). Slides of the same patient
    are merged into one patient graph before prediction.
    """

    def __init__(self, slides: Iterable[Slide], name: str = "cohort"):
        self.slides = list(slides)
        self.name = name
        seen: dict[str, list[Path]] = {}
        for s in self.slides:
            seen.setdefault(s.id, []).append(s.path)
        dup = {i: ps for i, ps in seen.items() if len(ps) > 1}
        if dup:
            shown = "\n".join(f"  {i}: " + ", ".join(str(p) for p in ps[:3]) for i, ps in sorted(dup.items())[:5])
            raise ValueError(
                f"{len(dup)} slide id(s) occur more than once (the id is the file name without its extension, so the "
                f"same name in two folders, or one slide saved in two formats, clashes). Remove the extra copy, use "
                f"--no-recursive, or list the slides in a CSV with unique slide_id values:\n{shown}")

    # ── constructors ───────────────────────────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_folder(cls, folder: str | Path, patient: str | None = None, recursive: bool = True,
                    extensions: Iterable[str] = SLIDE_EXTENSIONS) -> "Cohort":
        """All slides in `folder`. `patient` is a regex whose first group (or whole match) is the patient id,
        e.g. r"^(TCGA-\\w\\w-\\w{4})"; by default each slide is its own patient."""
        folder = Path(folder).absolute()     # absolute (not resolved) paths: workspaces work from any directory
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        exts = tuple(e.lower() for e in extensions)
        files = folder.rglob("*") if recursive else folder.iterdir()
        paths = sorted(p for p in files if p.is_file() and p.name.lower().endswith(exts) and not p.name.startswith("."))
        rx = re.compile(patient) if patient else None
        slides = []
        for p in paths:
            sid = slide_id(p)
            pid = sid
            if rx:
                m = rx.search(sid)
                if not m:
                    raise ValueError(f"patient pattern {patient!r} does not match slide {sid!r}")
                pid = m.group(1) if m.groups() else m.group(0)
            slides.append(Slide(sid, p, pid))
        return cls(slides, name=folder.name)

    @classmethod
    def from_csv(cls, csv: str | Path, slide_dir: str | Path | None = None, columns: dict[str, str] | None = None) -> "Cohort":
        """One row per slide. Columns: `slide_path` or `slide_id` (resolved inside `slide_dir`), optional `patient_id`.
        Use `columns={"my_col": "slide_id", ...}` to map other names."""
        df = pd.read_csv(csv).rename(columns=columns or {})
        if "slide_path" not in df and "slide_id" not in df:
            raise ValueError("CSV needs a slide_path or slide_id column")
        index = {}
        if "slide_path" not in df:
            if slide_dir is None:
                raise ValueError("pass slide_dir to resolve slide_id values")
            for p in Path(slide_dir).rglob("*"):
                if p.is_file():
                    index.setdefault(slide_id(p), p)
        slides = []
        for r in df.itertuples(index=False):
            row = r._asdict()
            if "slide_path" in row and isinstance(row["slide_path"], str):
                p = Path(row["slide_path"])
                if slide_dir is not None and not p.is_absolute():
                    p = Path(slide_dir) / p
            else:
                p = index.get(str(row["slide_id"]))
                if p is None:
                    raise FileNotFoundError(f"slide {row['slide_id']!r} not found in {slide_dir}")
            sid = str(row.get("slide_id") or slide_id(p))
            slides.append(Slide(sid, Path(p).absolute(), str(row.get("patient_id") or sid)))
        return cls(slides, name=Path(csv).stem)

    # ── views ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.slides)

    def __iter__(self) -> Iterator[Slide]:
        return iter(self.slides)

    def __getitem__(self, i):
        return self.slides[i]

    @property
    def slide_ids(self) -> list[str]:
        return [s.id for s in self.slides]

    @property
    def patients(self) -> dict[str, list[Slide]]:
        out: dict[str, list[Slide]] = {}
        for s in self.slides:
            out.setdefault(s.patient, []).append(s)
        return out

    def shard(self, index: int, count: int) -> "Cohort":
        """Every `count`-th patient starting at `index`, so array jobs split a cohort without overlap.
        Slides of one patient always stay in the same shard."""
        if not 0 <= index < count:
            raise ValueError("shard index must be in [0, count)")
        keep = {p for i, p in enumerate(sorted(self.patients)) if i % count == index}
        return Cohort([s for s in self.slides if s.patient in keep], name=f"{self.name}[{index}/{count}]")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"slide_id": s.id, "patient_id": s.patient, "slide_path": str(s.path)} for s in self.slides])

    def summary(self) -> str:
        n_pat = len(self.patients)
        multi = sum(len(v) > 1 for v in self.patients.values())
        exts = pd.Series([Path(s.path).suffix.lower() for s in self.slides]).value_counts().to_dict()
        return f"{self.name}: {len(self)} slides, {n_pat} patients ({multi} with several slides); formats {exts}"

    def __repr__(self) -> str:
        return f"Cohort({self.summary()})"
