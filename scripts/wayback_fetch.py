from __future__ import annotations

import time
from typing import Dict, Optional, Tuple
from urllib.parse import quote

import requests

# Identify your research use in the UA string
USER_AGENT = "ClimateSTANCE/1.1 (Research use; contact: saheerrahman98@gmail.com)"
HEADERS = {"User-Agent": USER_AGENT}

# Treat very small HTML as unusable
MIN_GOOD_LEN = 800


def _mid_month_timestamp(year: int, month: int) -> str:
    """
    Build YYYYMMDDhhmmss using the 15th at 12:00:00 to target the middle of the month.
    """
    return f"{year:04d}{month:02d}15" "120000"


def _get(url: str, timeout: int = 25, tries: int = 3, backoff: float = 0.75) -> requests.Response:
    """
    Wrapper around requests.get with UA header and simple retries.
    Raises the last exception if all retries fail.
    """
    last_exc: Optional[Exception] = None
    for i in range(tries):
        try:
            return requests.get(url, timeout=timeout, headers=HEADERS)
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff * (2 ** i))
    if last_exc:
        raise last_exc
    raise requests.RequestException("Unknown requests error")


def wayback_lookup(url: str, year: int, month: int) -> Tuple[Optional[str], Optional[str]]:
    """
    Ask Wayback for the closest archived snapshot to the target month.
    Returns (archive_url, snapshot_ts) or (None, None) if unavailable.

    API used (attributed): https://archive.org/help/wayback_api.php
    """
    if not url:
        return None, None
    ts = _mid_month_timestamp(year, month)
    api = f"https://archive.org/wayback/available?url={quote(url)}&timestamp={ts}"
    try:
        r = _get(api)
        r.raise_for_status()
        payload = r.json()
        closest = payload.get("archived_snapshots", {}).get("closest")
        if closest and closest.get("available"):
            return closest.get("url"), closest.get("timestamp")
        return None, None
    except requests.RequestException:
        return None, None


def fetch_live(url: str) -> Tuple[Optional[str], int]:
    """
    Try to fetch a live page. Returns (html, status_code).
    """
    if not url:
        return None, 0
    try:
        r = _get(url)
        if r.status_code == 200 and r.text and len(r.text) > MIN_GOOD_LEN:
            return r.text, r.status_code
        return None, r.status_code
    except requests.RequestException:
        return None, 0


def fetch_archived(archive_url: str) -> Tuple[Optional[str], int]:
    """
    Fetch an archived snapshot URL. Returns (html, status_code).
    """
    if not archive_url:
        return None, 0
    try:
        r = _get(archive_url)
        if r.status_code == 200 and r.text and len(r.text) > MIN_GOOD_LEN:
            return r.text, r.status_code
        return None, r.status_code
    except requests.RequestException:
        return None, 0


def collect_dual(url: str, year: int, month: int) -> Dict[str, str]:
    """
    Archive-first + keep-live-in-parallel.
    Returns both HTMLs (if present) and chooses which to use for the month.

    Keys returned:
      archive_html, status_archive, archive_url, snapshot_ts
      live_html,    status_live
      chosen_provenance: 'archive' | 'live' | 'none'
      reason: short note why chosen source was selected
    """
    out = {
        "archive_html": "", "status_archive": "",
        "archive_url": "", "snapshot_ts": "",
        "live_html": "", "status_live": "",
        "chosen_provenance": "none",
        "reason": ""
    }

    # 1) Archive first
    a_url, ts = wayback_lookup(url, year, month)
    if a_url:
        html_a, code_a = fetch_archived(a_url)
        out["archive_html"] = html_a or ""
        out["status_archive"] = str(code_a or "")
        out["archive_url"] = a_url
        out["snapshot_ts"] = ts or ""

    # 2) Live in parallel for redundancy
    html_l, code_l = fetch_live(url)
    out["live_html"] = html_l or ""
    out["status_live"] = str(code_l or "")

    # 3) Choose authoritative for this month (archive-first)
    if len(out["archive_html"]) >= MIN_GOOD_LEN:
        out["chosen_provenance"] = "archive"
        out["reason"] = "archive_ok"
    elif len(out["live_html"]) >= MIN_GOOD_LEN:
        out["chosen_provenance"] = "live"
        out["reason"] = "archive_missing_or_small_live_ok"
    else:
        out["chosen_provenance"] = "none"
        out["reason"] = "both_missing_or_small"

    return out