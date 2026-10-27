"""Download FIgLib wildfire smoke images from HPWREN.

License: FIgLib is CC BY-NC-ND 4.0. Do NOT redistribute the raw data.
Credit: Data courtesy of HPWREN, UC San Diego (https://www.hpwren.ucsd.edu/).

Usage:
    python scripts/fetch_data.py --list
    python scripts/fetch_data.py --sequence "<sequence-url>/" --dest data/raw
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

INDEX_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/index.html"
USER_AGENT = "ffsd-portfolio/0.1 (educational use)"


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "ignore")


def list_sequences(index_url: str = INDEX_URL) -> list[str]:
    html = _get(index_url)
    hrefs = re.findall(r'href="([^"]+/)"', html)
    return sorted(set(hrefs))


def download_sequence(seq_url: str, dest: str | Path) -> int:
    out_dir = Path(dest)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = _get(seq_url)
    jpgs = re.findall(r'href="([^"]+\.jpg)"', html, flags=re.IGNORECASE)

    count = 0
    for name in jpgs:
        url = urljoin(seq_url, name)
        target = out_dir / name
        if target.exists():
            continue
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            target.write_bytes(response.read())
        count += 1
        print(f"downloaded {target.name}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FIgLib sequences (CC BY-NC-ND 4.0)")
    parser.add_argument("--list", action="store_true", help="list available sequences")
    parser.add_argument("--sequence", help="sequence directory URL to download")
    parser.add_argument("--dest", default="data/raw", help="destination directory")
    args = parser.parse_args()

    if args.list:
        for seq in list_sequences():
            print(seq)
        return

    if args.sequence:
        n = download_sequence(args.sequence, args.dest)
        print(f"done, {n} files")

    print("NOTE: FIgLib is CC BY-NC-ND 4.0. Do not redistribute raw data.")
    print("Credit: Data courtesy of HPWREN, UC San Diego (https://www.hpwren.ucsd.edu/).")


if __name__ == "__main__":
    main()
