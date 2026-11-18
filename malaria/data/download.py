"""Stage 1: download the published archives from Harvard Dataverse and verify their checksums.

The file list is read from the Dataverse API at run time, so nothing about the dataset's
file names is hard-coded. Each file's MD5 comes from the same API and is checked after download.

If the download cannot run from your network, download the files by hand from
https://doi.org/10.7910/DVN/VEADSE into the raw data folder and re-run: files already present are
checked and kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from malaria import config

UA = {"User-Agent": "malaria-capstone-pipeline/1.0 (student research; python-requests)"}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_listing(payload: dict) -> list[dict]:
    """Turn the Dataverse dataset JSON into a flat list of files."""
    files = []
    for entry in payload["data"]["latestVersion"]["files"]:
        data_file = entry["dataFile"]
        checksum = data_file.get("checksum") or {}
        md5 = data_file.get("md5") or (checksum.get("value") if checksum.get("type", "").upper() == "MD5" else None)
        files.append({
            "id": data_file["id"],
            "name": data_file["filename"],
            "size": data_file.get("filesize"),
            "md5": md5,
            "dir": entry.get("directoryLabel", ""),
            "description": entry.get("description", ""),
        })
    return files


def select(files: list[dict], include: list[str], exclude: list[str]) -> list[dict]:
    chosen = []
    for f in files:
        name = f["name"]
        if include and not any(re.search(p, name) for p in include):
            continue
        if any(re.search(p, name) for p in exclude):
            continue
        chosen.append(f)
    return chosen


def list_files(cfg: config.Cfg) -> list[dict]:
    import requests

    ds = cfg["dataset"]
    url = f"{ds['dataverse_url'].rstrip('/')}/api/datasets/:persistentId/"
    resp = requests.get(url, params={"persistentId": ds["doi"]}, headers=UA, timeout=60)
    resp.raise_for_status()
    return parse_listing(resp.json())


def download_one(cfg: config.Cfg, f: dict, dest: Path, retries: int = 5) -> str:
    """Download one file with resume. Returns 'kept', 'downloaded' or raises."""
    import requests

    if dest.exists() and (f["size"] is None or dest.stat().st_size == f["size"]):
        if not f["md5"] or md5sum(dest) == f["md5"]:
            return "kept"
        dest.unlink()
    url = f"{cfg['dataset']['dataverse_url'].rstrip('/')}/api/access/datafile/{f['id']}"
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code == 416:  # nothing left to fetch
                    break
                resp.raise_for_status()
                mode = "ab" if have and resp.status_code == 206 else "wb"
                with part.open(mode) as out:
                    for block in resp.iter_content(1 << 20):
                        out.write(block)
            if f["size"] is None or part.stat().st_size >= f["size"]:
                break
        except Exception as exc:  # network errors: wait and resume
            if attempt == retries:
                raise
            print(f"  attempt {attempt} failed ({exc}); resuming in {5 * attempt}s")
            time.sleep(5 * attempt)
    if f["md5"]:
        got = md5sum(part)
        if got != f["md5"]:
            raise RuntimeError(f"checksum mismatch for {f['name']}: expected {f['md5']}, got {got}")
    part.replace(dest)
    return "downloaded"


def run(cfg: config.Cfg, list_only: bool = False) -> list[dict]:
    raw = cfg.path("raw", make=True)
    try:
        files = list_files(cfg)
    except Exception as exc:
        present = [p.name for p in raw.iterdir() if p.is_file() and p.name != "manifest.json"]
        if present:
            print(f"[download] could not reach Dataverse ({exc}); continuing with {len(present)} file(s) already in {raw}")
            return []
        raise SystemExit(
            f"[download] could not reach Dataverse ({exc}).\n"
            f"Download the thick-smear archives by hand from https://doi.org/10.7910/DVN/VEADSE into {raw} and re-run."
        )
    chosen = select(files, cfg["dataset"]["include"], cfg["dataset"]["exclude"])
    total = sum(f["size"] or 0 for f in chosen)
    print(f"[download] {len(chosen)} of {len(files)} files selected, {total / 1e9:.1f} GB")
    for f in chosen:
        print(f"  {f['name']:<50} {(f['size'] or 0) / 1e6:>9.1f} MB  md5={f['md5']}")
    if list_only:
        return chosen
    manifest = []
    for f in chosen:
        status = download_one(cfg, f, raw / f["name"])
        print(f"  {status:<10} {f['name']}")
        manifest.append({**f, "status": status, "verified": bool(f["md5"])})
    (raw / "manifest.json").write_text(json.dumps({"doi": cfg["dataset"]["doi"], "files": manifest}, indent=2))
    return manifest


def main() -> None:
    parser = config.add_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--list", action="store_true", help="print the selected files and stop")
    args = parser.parse_args()
    run(config.from_args(args), list_only=args.list)


if __name__ == "__main__":
    main()
