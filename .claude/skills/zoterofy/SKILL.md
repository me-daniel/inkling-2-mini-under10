---
name: zoterofy
description: Keep a project's sources in a ledger (Markdown, one line per source), verify each one against its DOI, arXiv, ISBN or URL record, add the verified ones to the project's Zotero collection with a pinned citation key, pull refs.bib from Zotero and turn "TODO cite" markers into real citations. Use for "/zoterofy", "add these sources to Zotero", "zoterofy this", a pasted sources block from a Claude app research session, new references found during research, or a paper with TODO cite markers or undefined citations.
argument-hint: "[pasted sources block | verify | add | pull | cite]"
---

# /zoterofy

Sources reach the paper in four steps, and the ledger records the step each one has reached. Nothing goes to Zotero until its record has been checked, and nothing reaches the bib except through Zotero.

```
[ ] candidate  -> verify ->  [v] verified  -> add ->  [x] in Zotero  -> pull -> refs.bib -> cite
[?] doubtful (kept with its reason, never added)
```

**Project settings.** The "Bibliography" line under "Project specifics" in `CLAUDE.md` names the ledger, the bib file and the pull command. If it is missing, ask once and add it, for example: `- Bibliography: ledger docs/sources.md; refs paper/refs.bib via make -C paper refs (Zotero collection in ZOTERO_COLLECTION)`.

**Script.** `python3 -I .claude/skills/zoterofy/zotero.py <command>`, written `zotero.py` below (stdlib only; spell the command out in full, since zsh does not word-split a `$Z` variable). It reads `ZOTERO_API_KEY`, `ZOTERO_LIBRARY` and `ZOTERO_COLLECTION` from the environment or from `.env` at the git root by itself. Never read, print or `source` `.env` yourself.

## Ledger line

```
- [ ] `frederick2005cognitive` doi:10.1257/089533005775196732 · Frederick 2005, "Cognitive Reflection and Decision Making" · for: the CRT and its three items (intro)
```

- Citation key: first author's family name, year, first content word of the title, all lowercase ASCII.
- Id: `doi:` when the source has a DOI (prefer the published version), else `arxiv:`, `isbn:` (books) or `url:` (model cards, docs, web pages).
- `Author Year, "Title"` as on the record. The script compares them with the record, so they are what the check checks.
- `for:` the claim it supports and where. Free text after it is kept as is.

## Steps

1. **Collect.** Append candidates as `[ ]` lines under a fitting heading of the ledger.
   - From a Claude app session: the user pastes the block that the prompt below produces. Keep its `[?]` lines.
   - From research in this repo or from `% TODO cite` markers: only sources whose id you saw on the source's own page. Never write an id, title or year from memory. If you can't find the id, add the line as `[?]` with "id not found".
   - Skip a key or id that is already in the ledger. Mention it instead.
2. **Verify.** Run `zotero.py verify LEDGER --dry-run --abstracts`.
   - `OK` means the id resolves and its title, first author and year match the line. It does not mean the source supports the claim. Read the abstract (or open the page for `url:` lines) against the `for:` note. Change a line that doesn't hold up to `[?]` with the reason.
   - `MISMATCH` shows the record. Fix the line only when the record is clearly the same work (a preprint title, a year off between arXiv and venue). Note the difference after `for:`. Otherwise mark it `[?]`.
   - `ERROR` is usually a network problem or a wrong id. Rerun, then check the id on the source's page.
   - Then run `zotero.py verify LEDGER` for real. It marks every `OK` candidate `[v]`.
3. **Add.** `zotero.py check` confirms the key, the collection and write access. Then run `zotero.py add LEDGER --dry-run` and read the items: type, venue, authors, publisher. Records can be wrong too. Then `zotero.py add LEDGER`.
   - Each new item goes into the collection with `Citation Key: <key>` in Extra. Zotero's BibTeX export uses that line as the entry key, so keys stay stable.
   - An item already in the library gets the collection and the key added (`EXISTING`). If it already has a different key, it stays as it is (`KEPT`). Tell the user.
   - The script marks each added line `[x] … · zotero:<item key>` straight away, so a rerun after a crash continues where it stopped.
4. **Pull.** Use the project's pull command, or `zotero.py pull OUT.bib --ledger LEDGER` to pull only the ledger's `[x]` items. The ledger form also lists any ledger key missing from the bib. Never edit the bib by hand.
5. **Cite.** Replace each `% TODO cite ... \citep{k}` with `\citep{k}`, or the project's citation command, but only for keys present in the bib. Build, and fix every undefined citation. Leave the TODO for any key not yet in the bib, and say why.

## Prompt for the Claude app

Give the user this to paste at the end of a research session:

> End with a sources block in a code fence. One line per source you actually opened, in exactly this form:
> `- [ ] `familyYYYYword` doi:10.xxxx/yyyy · Family YYYY, "Exact title" · for: the claim it supports`
> Use `doi:` when the page shows a DOI, else `arxiv:<id>`, `isbn:<isbn>` for books, or `url:<url>`. Copy ids and titles from the source's page. Leave out any source whose id you did not see. List sources you are unsure about the same way but with `[?]` instead of `[ ]`, plus the reason.

## Rules

- No source is added from memory, and no record is fixed by guessing. If it can't be checked, it stays `[?]`.
- The write key belongs in `.env` as `ZOTERO_API_KEY`, with write access set at zotero.org/settings/keys. If `check` says `write NO`, ask the user to enable it there. Never ask them to paste the key into the chat.
- `add` writes to the user's Zotero library. A rerun is safe (existing items are found, not duplicated), but deleting items is the user's job, in Zotero.
