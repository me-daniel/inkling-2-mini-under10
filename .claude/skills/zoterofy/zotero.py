#!/usr/bin/env python3
"""zoterofy: verify the sources in a ledger, add them to a Zotero collection, pull refs.bib.

A ledger is any Markdown file. Each source is one line:
  - [ ] `frederick2005cognitive` doi:10.1257/089533005775196732 · Frederick 2005, "Cognitive reflection and decision making" · for: CRT origin
Ids: doi:<doi>, arxiv:<id>, isbn:<isbn>, url:<url>. Status:
  [ ] candidate -> verify -> [v] verified -> add -> [x] in Zotero (" · zotero:<item key>" appended)
  [?] doubtful (kept, never added; give the reason on the line)

  zotero.py check                        key, library, collection, write access
  zotero.py verify LEDGER [--dry-run]    resolve each [ ] id, compare title/author/year, mark [v]
  zotero.py add LEDGER [--dry-run]       create each [v] item in the collection, key pinned in Extra, mark [x]
  zotero.py pull OUT.bib [--ledger L]    BibTeX of the collection (or only L's [x] items); lists L's keys missing

ZOTERO_API_KEY, ZOTERO_LIBRARY (users/<id> or groups/<id>) and ZOTERO_COLLECTION come from the environment,
else from .env at the git root. Values are never printed. Stdlib only.
"""

import argparse
import datetime
import difflib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "https://api.zotero.org"
UA = "zoterofy/1.0 (python-urllib)"
LINE = re.compile(r"^(\s*- \[)([ vx?])(\] `([^`]+)` (doi|arxiv|isbn|url):(\S+))(.*)$")


# ---------- config and HTTP ----------


def env() -> dict:
    keys = ("ZOTERO_API_KEY", "ZOTERO_LIBRARY", "ZOTERO_COLLECTION")
    vals = {k: os.environ.get(k, "") for k in keys}
    if not all(vals.values()):
        try:
            root = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
            ).stdout.strip()
            with open(os.path.join(root, ".env")) as f:
                for line in f:  # KEY=v, KEY = v, export KEY=v, KEY:=v (make syntax)
                    m = re.match(r"\s*(?:export\s+)?(\w+)\s*:?=\s*(.*?)\s*$", line)
                    if m and m.group(1) in keys and not vals[m.group(1)]:
                        vals[m.group(1)] = m.group(2).split(" #")[0].strip().strip("'\"")
        except (OSError, subprocess.CalledProcessError):
            pass
    url_key = re.search(r"/collections/([A-Z0-9]{8})", vals["ZOTERO_COLLECTION"])
    if url_key:  # a pasted web-library URL
        vals["ZOTERO_COLLECTION"] = url_key.group(1)
    return vals


def http(url, data=None, headers=None, method=None):
    """-> (status, headers, body text). Raises RuntimeError with a short message on HTTP errors.
    Reads (GET) are retried on timeouts, 429 and 5xx; writes never are, so an item is never created twice."""
    h = {"User-Agent": UA, **(headers or {})}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        h["Content-Type"] = "application/json"
    tries = 1 if (method or ("POST" if data is not None else "GET")) != "GET" else 4
    for n in range(tries):
        if "arxiv.org" in url:
            time.sleep(3)  # arXiv asks for 3 s between API calls
        req = urllib.request.Request(url, data=body, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.headers, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if (e.code == 429 or e.code >= 500) and n < tries - 1:
                time.sleep(5 * (n + 1))
                continue
            raise RuntimeError(f"HTTP {e.code} for {url.split('?')[0]}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if n < tries - 1:
                time.sleep(5 * (n + 1))
                continue
            raise RuntimeError(f"{url.split('?')[0]}: {e}") from None


def zot(cfg, path, data=None, method=None, extra=None):
    h = {"Zotero-API-Key": cfg["ZOTERO_API_KEY"], "Zotero-API-Version": "3", **(extra or {})}
    return http(f"{API}/{path}", data, h, method)


def need(cfg, *keys):
    missing = [k for k in keys if not cfg[k]]
    if missing:
        sys.exit(f"set {', '.join(missing)} (environment or .env at the git root)")


# ---------- resolvers: id -> common metadata ----------

CSL_TYPES = {
    "article-journal": "journalArticle",
    "journal-article": "journalArticle",
    "paper-conference": "conferencePaper",
    "proceedings-article": "conferencePaper",
    "book": "book",
    "chapter": "bookSection",
    "posted-content": "preprint",
    "article": "preprint",
    "report": "report",
    "dataset": "dataset",
    "webpage": "webpage",
    "thesis": "thesis",
}


def date_of(parts):
    p = (parts or {}).get("date-parts", [[]])[0]
    return "-".join(f"{int(x):02d}" if i else str(x) for i, x in enumerate(p) if x)


def from_doi(doi):
    _, _, body = http(
        f"https://doi.org/{urllib.parse.quote(doi, safe='/')}",
        headers={"Accept": "application/vnd.citationstyles.csl+json"},
    )
    c = json.loads(body)
    title = c.get("title") or ""
    title = title[0] if isinstance(title, list) else title
    sub = c.get("subtitle") or []
    if isinstance(sub, list) and sub:
        title = f"{title}: {sub[0]}"
    venue = c.get("container-title") or ""
    venue = venue[0] if isinstance(venue, list) and venue else venue or ""
    return {
        "type": CSL_TYPES.get(c.get("type"), "journalArticle"),
        "title": title,
        "authors": [
            (a.get("family", ""), a.get("given", ""), a.get("literal", "")) for a in c.get("author", [])
        ],
        "editors": [
            (a.get("family", ""), a.get("given", ""), a.get("literal", "")) for a in c.get("editor", [])
        ],
        "date": date_of(c.get("issued") or c.get("published-print") or c.get("published")),
        "venue": venue,
        "volume": c.get("volume", ""),
        "issue": c.get("issue", ""),
        "pages": c.get("page", ""),
        "publisher": c.get("publisher", ""),
        "place": c.get("publisher-place", ""),
        "DOI": c.get("DOI", doi),
        "url": c.get("URL", f"https://doi.org/{doi}"),
        "abstract": re.sub(r"<[^>]+>", "", c.get("abstract", "")).strip(),
    }


def from_arxiv(aid):
    _, _, body = http(f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid)}")
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    e = ET.fromstring(body).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        raise RuntimeError(f"arXiv has no record {aid}")
    names = [n.text.strip() for n in e.findall("a:author/a:name", ns)]
    bare = re.sub(r"v\d+$", "", aid)
    return {
        "type": "preprint",
        "title": " ".join(e.find("a:title", ns).text.split()),
        "authors": [(n.rsplit(" ", 1)[-1], n.rsplit(" ", 1)[0] if " " in n else "", "") for n in names],
        "editors": [],
        "date": e.find("a:published", ns).text[:10],
        "venue": "arXiv",
        "publisher": "arXiv",
        "DOI": f"10.48550/arXiv.{bare}",
        "url": f"https://arxiv.org/abs/{bare}",
        "archiveID": f"arXiv:{bare}",
        "abstract": " ".join((e.find("a:summary", ns).text or "").split()),
    }


def from_isbn(isbn):
    _, _, body = http(f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data")
    b = json.loads(body).get(f"ISBN:{isbn}")
    if not b:
        raise RuntimeError(f"Open Library has no record for ISBN {isbn}")
    year = re.search(r"\d{4}", b.get("publish_date", ""))
    return {
        "type": "book",
        "title": b.get("title", "") + (f": {b['subtitle']}" if b.get("subtitle") else ""),
        "authors": [
            (a["name"].rsplit(" ", 1)[-1], a["name"].rsplit(" ", 1)[0], "") for a in b.get("authors", [])
        ],
        "editors": [],
        "date": year.group(0) if year else "",
        "venue": "",
        "publisher": ", ".join(p["name"] for p in b.get("publishers", [])),
        "place": ", ".join(p["name"] for p in b.get("publish_places", [])),
        "ISBN": isbn,
        "url": "",
        "abstract": "",
    }


def from_url(url):
    _, _, body = http(url)
    m = re.search(r"<title[^>]*>(.*?)</title>", body, re.DOTALL | re.IGNORECASE)
    return {
        "type": "webpage",
        "title": " ".join(m.group(1).split()) if m else "",
        "authors": [],
        "editors": [],
        "date": "",
        "venue": "",
        "url": url,
        "abstract": "",
        "page_title": True,
    }


RESOLVE = {"doi": from_doi, "arxiv": from_arxiv, "isbn": from_isbn, "url": from_url}


# ---------- ledger ----------


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def parse(path):
    with open(path) as f:
        lines = f.read().split("\n")
    entries = []
    for i, line in enumerate(lines):
        m = LINE.match(line)
        if m:
            rest = m.group(7)
            head = rest.split('"', 1)[0]
            title = re.search(r'"([^"]+)"', rest)
            year = re.search(r"\b(1[89]|20)\d{2}\b", head)
            zk = re.search(r"zotero:([A-Z0-9]{8})", rest)
            entries.append(
                {
                    "i": i,
                    "status": m.group(2),
                    "key": m.group(4),
                    "kind": m.group(5),
                    "id": m.group(6),
                    "head": head,
                    "title": title.group(1) if title else "",
                    "year": year.group(0) if year else "",
                    "zkey": zk.group(1) if zk else "",
                }
            )
    return lines, entries


def mark(lines, e, status, zkey=""):
    m = LINE.match(lines[e["i"]])
    rest = m.group(7)
    if zkey and "zotero:" not in rest:
        rest += f" · zotero:{zkey}"
    lines[e["i"]] = m.group(1) + status + m.group(3) + rest


def save(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines))


def cite(meta):
    a = meta["authors"][0] if meta["authors"] else ("", "", "")
    who = a[0] or a[2] or "?"
    return f'{who} {meta["date"][:4]}, "{meta["title"]}"' + (
        f" ({meta['venue']})" if meta.get("venue") else ""
    )


def check_entry(e, meta):
    """-> list of problems; empty means the ledger line matches the record."""
    if not e["title"]:
        return ['no "title" on the ledger line to compare with']
    t, r = norm(e["title"]), norm(meta["title"])
    if meta.get("page_title"):
        ok = t and (t in r or difflib.SequenceMatcher(None, t, r).ratio() >= 0.9)
        return [] if ok else [f"page title is {meta['title']!r}"]
    probs = []
    if difflib.SequenceMatcher(None, t, r).ratio() < 0.9:
        probs.append(f"title is {meta['title']!r}")
    if meta["authors"]:
        fam = norm(meta["authors"][0][0] or meta["authors"][0][2]).split()
        if fam and fam[-1] not in norm(e["head"]).split():
            probs.append(f"first author is {meta['authors'][0][0] or meta['authors'][0][2]!r}")
    if e["year"] and meta["date"][:4] and e["year"] != meta["date"][:4]:
        probs.append(f"year is {meta['date'][:4]}")
    return probs


# ---------- Zotero items ----------

_templates = {}


def template(item_type):
    if item_type not in _templates:
        _, _, body = http(f"{API}/items/new?itemType={item_type}")
        _templates[item_type] = json.loads(body)
    return _templates[item_type]


def zotero_item(e, meta, collection):
    tpl = template(meta["type"])
    v = meta.get("venue", "")
    fields = {
        "title": e["title"] if meta.get("page_title") else meta["title"],
        "date": meta.get("date", ""),
        "url": meta.get("url", ""),
        "DOI": meta.get("DOI", ""),
        "ISBN": meta.get("ISBN", ""),
        "abstractNote": meta.get("abstract", ""),
        "volume": meta.get("volume", ""),
        "issue": meta.get("issue", ""),
        "pages": meta.get("pages", ""),
        "publisher": meta.get("publisher", ""),
        "institution": meta.get("publisher", ""),
        "place": meta.get("place", ""),
        "publicationTitle": v,
        "proceedingsTitle": v,
        "bookTitle": v,
        "repository": v,
        "archiveID": meta.get("archiveID", ""),
        "accessDate": datetime.date.today().isoformat() if meta["type"] == "webpage" else "",
        "extra": f"Citation Key: {e['key']}",
    }
    item = {k: val for k, val in fields.items() if k in tpl and val}
    item["itemType"] = meta["type"]
    item["collections"] = [collection]
    people = [("author", p) for p in meta["authors"]] + [("editor", p) for p in meta.get("editors", [])]
    item["creators"] = [
        {"creatorType": role, "name": lit}
        if lit and not fam
        else {"creatorType": role, "lastName": fam, "firstName": given}
        for role, (fam, given, lit) in people
    ]
    return item


def find_existing(cfg, meta):
    q = urllib.parse.quote(meta["title"][:80])
    _, _, body = zot(cfg, f"{cfg['ZOTERO_LIBRARY']}/items?q={q}&limit=25&itemType=-attachment")
    for it in json.loads(body):
        d = it["data"]
        same_doi = meta.get("DOI") and d.get("DOI", "").lower() == meta["DOI"].lower()
        if same_doi or norm(d.get("title", "")) == norm(meta["title"]):
            return d
    return None


# ---------- commands ----------


def cmd_check(a):
    cfg = env()
    need(cfg, "ZOTERO_API_KEY")
    _, _, body = zot(cfg, "keys/current")
    me = json.loads(body)
    acc = me.get("access", {})
    print(f"key belongs to users/{me.get('userID')}")
    if not re.fullmatch(r"(users|groups)/\d+", cfg["ZOTERO_LIBRARY"]):
        sys.exit(f"ZOTERO_LIBRARY must look like users/<numeric id>, e.g. users/{me.get('userID')}")
    need(cfg, "ZOTERO_COLLECTION")
    kind, _, lid = cfg["ZOTERO_LIBRARY"].partition("/")
    scope = (
        acc.get("user", {})
        if kind == "users"
        else acc.get("groups", {}).get(lid, acc.get("groups", {}).get("all", {}))
    )
    _, _, body = zot(cfg, f"{cfg['ZOTERO_LIBRARY']}/collections/{cfg['ZOTERO_COLLECTION']}")
    c = json.loads(body)
    print(
        f"library {cfg['ZOTERO_LIBRARY']}: read {'yes' if scope.get('library') else 'no'}, "
        f"write {'yes' if scope.get('write') else 'NO (edit the key at zotero.org/settings/keys)'}"
    )
    print(f"collection {c['data']['name']!r}: {c['meta'].get('numItems', '?')} items")


def cmd_verify(a):
    lines, entries = parse(a.ledger)
    todo = [e for e in entries if e["status"] == " "]
    print(f"{len(todo)} candidate(s) of {len(entries)} ledger entries")
    n_ok = 0
    for e in todo:
        try:
            meta = RESOLVE[e["kind"]](e["id"])
        except (RuntimeError, OSError, ValueError, ET.ParseError) as err:
            print(f"ERROR     {e['key']}: {err}")
            continue
        probs = check_entry(e, meta)
        if probs:
            print(f"MISMATCH  {e['key']}: {'; '.join(probs)}\n          record: {cite(meta)}")
            continue
        n_ok += 1
        print(f"OK        {e['key']}: {cite(meta)}")
        if a.abstracts and meta.get("abstract"):
            print(f"          abstract: {meta['abstract'][:600]}")
        if not a.dry_run:
            mark(lines, e, "v")
    if not a.dry_run:
        save(a.ledger, lines)
    print(f"{n_ok} matched" + (" (dry run, ledger unchanged)" if a.dry_run else ", marked [v]"))


def cmd_add(a):
    cfg = env()
    need(cfg, "ZOTERO_LIBRARY", "ZOTERO_COLLECTION", *([] if a.dry_run else ["ZOTERO_API_KEY"]))
    lines, entries = parse(a.ledger)
    todo = [e for e in entries if e["status"] == "v"]
    print(f"{len(todo)} verified entr{'y' if len(todo) == 1 else 'ies'} to add")
    for e in todo:
        try:
            meta = RESOLVE[e["kind"]](e["id"])
            item = zotero_item(e, meta, cfg["ZOTERO_COLLECTION"])
            if a.dry_run:
                print(json.dumps(item, indent=1, ensure_ascii=False))
                continue
            d = find_existing(cfg, meta)
            if d:
                extra = d.get("extra", "")
                ck = re.search(r"(?im)^citation key:\s*(\S+)", extra)
                patch = {}
                if cfg["ZOTERO_COLLECTION"] not in d.get("collections", []):
                    patch["collections"] = d.get("collections", []) + [cfg["ZOTERO_COLLECTION"]]
                if not ck:
                    patch["extra"] = (extra + "\n" if extra else "") + f"Citation Key: {e['key']}"
                elif ck.group(1) != e["key"]:
                    print(
                        f"KEPT      {e['key']}: already in Zotero as {d['key']} with key {ck.group(1)}; "
                        f"fix the ledger key or the Extra field"
                    )
                    continue
                if patch:
                    zot(
                        cfg,
                        f"{cfg['ZOTERO_LIBRARY']}/items/{d['key']}",
                        patch,
                        "PATCH",
                        {"If-Unmodified-Since-Version": str(d["version"])},
                    )
                print(
                    f"EXISTING  {e['key']}: {d['key']}" + (f" (updated {', '.join(patch)})" if patch else "")
                )
                zkey = d["key"]
            else:
                _, _, body = zot(cfg, f"{cfg['ZOTERO_LIBRARY']}/items", [item], "POST")
                res = json.loads(body)
                if res.get("failed"):
                    print(f"FAILED    {e['key']}: {json.dumps(res['failed'])[:300]}")
                    continue
                zkey = res["successful"]["0"]["key"]
                print(f"ADDED     {e['key']}: {zkey}  {cite(meta)}")
            mark(lines, e, "x", zkey)
            save(a.ledger, lines)  # after each item, so a crash never loses a mark
        except (RuntimeError, OSError, ValueError, ET.ParseError) as err:
            print(f"ERROR     {e['key']}: {err}")


def cmd_pull(a):
    cfg = env()
    need(cfg, "ZOTERO_API_KEY", "ZOTERO_LIBRARY", *([] if a.ledger else ["ZOTERO_COLLECTION"]))
    want = []
    if a.ledger:
        _, entries = parse(a.ledger)
        want = [e for e in entries if e["status"] == "x" and e["zkey"]]
    chunks, start = [], 0
    while True:
        if a.ledger:
            keys = ",".join(e["zkey"] for e in want[start : start + 50])
            if not keys:
                break
            path = f"{cfg['ZOTERO_LIBRARY']}/items?itemKey={keys}&format=bibtex&limit=50"
        else:
            path = (
                f"{cfg['ZOTERO_LIBRARY']}/collections/{cfg['ZOTERO_COLLECTION']}/items/top"
                f"?format=bibtex&limit=100&start={start}"
            )
        _, _, body = zot(cfg, path)
        if not body.strip():
            break
        chunks.append(body.strip())
        start += 50 if a.ledger else 100
    bib = "\n\n".join(chunks) + "\n"
    with open(a.out + ".tmp", "w") as f:
        f.write(bib)
    os.replace(a.out + ".tmp", a.out)
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib, re.MULTILINE))
    print(f"{a.out}: {len(keys)} entries")
    if a.ledger:
        missing = [e["key"] for e in want if e["key"] not in keys]
        print(
            "every ledger key is in the bib"
            if not missing
            else f"ledger keys missing from the bib: {', '.join(missing)}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(f=cmd_check)
    v = sub.add_parser("verify")
    v.add_argument("ledger")
    v.add_argument("--dry-run", action="store_true")
    v.add_argument("--abstracts", action="store_true", help="print each record's abstract")
    v.set_defaults(f=cmd_verify)
    ad = sub.add_parser("add")
    ad.add_argument("ledger")
    ad.add_argument("--dry-run", action="store_true", help="print the Zotero items, send nothing")
    ad.set_defaults(f=cmd_add)
    pl = sub.add_parser("pull")
    pl.add_argument("out")
    pl.add_argument("--ledger", help="pull only this ledger's [x] items and report missing keys")
    pl.set_defaults(f=cmd_pull)
    a = p.parse_args()
    try:
        a.f(a)
    except RuntimeError as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
