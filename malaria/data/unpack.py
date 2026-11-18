"""Stage 2: unpack the raw archives into one folder per collection site.

Multi-part RAR sets are extracted once, from their first part. RAR needs an external tool:
7-Zip (Windows: https://www.7-zip.org), `unrar`, or `bsdtar`.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from malaria import config

PART = re.compile(r"\.part0*(\d+)\.rar$", re.I)
WINDOWS_7Z = [r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"]


def site_of(name: str, sites: list[str]) -> str:
    lowered = name.lower()
    for site in sites:
        if site.lower() in lowered:
            return site.lower()
    return "unknown"


def is_first_volume(name: str) -> bool:
    """True for single archives and for the first part of a multi-part RAR set."""
    match = PART.search(name)
    return match is None or int(match.group(1)) == 1


def archive_stem(name: str) -> str:
    stem = PART.sub("", name)
    return re.sub(r"\.(rar|zip|7z|tar\.gz|tgz|tar)$", "", stem, flags=re.I)


def find_rar_tool(multipart: bool = False) -> list[str] | None:
    """Prefer 7-Zip and unrar. Windows' built-in tar.exe (bsdtar) opens single RAR files but cannot
    follow a multi-part set, so it is used only as a last resort for single archives."""
    for exe in ("7z", "7zz", "7za"):
        if shutil.which(exe):
            return [shutil.which(exe), "x", "-y", "-o{dest}", "{src}"]
    for candidate in WINDOWS_7Z:
        if Path(candidate).exists():
            return [candidate, "x", "-y", "-o{dest}", "{src}"]
    if shutil.which("unrar"):
        return [shutil.which("unrar"), "x", "-o+", "{src}", "{dest}/"]
    if shutil.which("unar"):
        return [shutil.which("unar"), "-f", "-o", "{dest}", "{src}"]
    if multipart:
        return None
    for exe in ("bsdtar", "tar"):
        if shutil.which(exe):
            return [shutil.which(exe), "-xf", "{src}", "-C", "{dest}"]
    return None


def _safe_members(names: list[str], dest: Path) -> None:
    root = dest.resolve()
    for name in names:
        target = (dest / name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"archive member escapes the destination folder: {name}")


def extract(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    lowered = src.name.lower()
    if lowered.endswith(".zip"):
        with zipfile.ZipFile(src) as zf:
            _safe_members(zf.namelist(), dest)
            zf.extractall(dest)
    elif lowered.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(src) as tf:
            _safe_members(tf.getnames(), dest)
            tf.extractall(dest)
    elif lowered.endswith((".rar", ".7z")):
        multipart = PART.search(src.name) is not None
        tool = find_rar_tool(multipart)
        if tool is None:
            raise SystemExit("[unpack] no tool can open this archive. Install 7-Zip (https://www.7-zip.org, "
                             "or `winget install 7zip.7zip`) and re-run; Windows' tar.exe cannot read multi-part RAR sets.")
        cmd = [part.format(src=str(src), dest=str(dest)) for part in tool]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    else:
        raise ValueError(f"not an archive: {src.name}")


def is_archive(name: str) -> bool:
    return name.lower().endswith((".rar", ".zip", ".7z", ".tar", ".tar.gz", ".tgz"))


def run(cfg: config.Cfg) -> None:
    raw, interim = cfg.path("raw"), cfg.path("interim", make=True)
    sites = cfg["dataset"]["sites"]
    excluded = cfg["dataset"]["exclude"]
    files = sorted(p for p in raw.iterdir() if p.is_file() and p.name != "manifest.json" and not p.name.endswith(".part"))
    if not files:
        raise SystemExit(f"[unpack] nothing in {raw}; run the download stage first.")
    for src in files:
        if any(re.search(p, src.name) for p in excluded):
            continue
        site = site_of(src.name, sites)
        if not is_archive(src.name):
            loose = interim / site / "_loose"
            loose.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, loose / src.name)
            continue
        if not is_first_volume(src.name):
            continue
        dest = interim / site / archive_stem(src.name)
        marker = dest / ".unpacked"
        if marker.exists():
            print(f"[unpack] kept      {src.name}")
            continue
        if dest.exists():
            print(f"[unpack] removing the incomplete folder from an earlier attempt: {dest}")
            shutil.rmtree(dest)
        print(f"[unpack] extracting {src.name} -> {dest}")
        extract(src, dest)
        marker.write_text(src.name)
    for site_dir in sorted(p for p in interim.iterdir() if p.is_dir()):
        count = sum(1 for _ in site_dir.rglob("*") if _.is_file())
        print(f"[unpack] {site_dir.name}: {count} files")


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    run(config.from_args(parser.parse_args()))


if __name__ == "__main__":
    main()
