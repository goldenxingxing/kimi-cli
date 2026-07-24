#!/usr/bin/env python3
"""Extract embedded media from .docx and build an index with captions/context.

Usage:
    python extract_docx_media.py input.docx --out-dir ./media_dump
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


DOCX_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract media from a .docx file")
    parser.add_argument("docx", help="Input .docx file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_media(docx_path: str, media_dir: Path) -> list[dict]:
    """Copy files from word/media/ to media_dir and return metadata."""
    media_items: list[dict] = []
    with zipfile.ZipFile(docx_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("word/media/") and not name.endswith("/"):
                data = zf.read(name)
                filename = Path(name).name
                ext = Path(name).suffix.lower()
                out_path = media_dir / filename
                out_path.write_bytes(data)
                media_items.append({
                    "archive_path": name,
                    "filename": filename,
                    "extension": ext,
                    "saved_path": str(out_path),
                    "size_bytes": len(data),
                })
    return media_items


def build_rid_to_target(docx_path: str) -> dict[str, str]:
    """Map relationship IDs to target filenames inside word/_rels/document.xml.rels."""
    rid_map: dict[str, str] = {}
    rels_path = "word/_rels/document.xml.rels"
    with zipfile.ZipFile(docx_path, "r") as zf:
        if rels_path not in zf.namelist():
            return rid_map
        data = zf.read(rels_path)
    root = ET.fromstring(data)
    for rel in root.findall("rel:Relationship", DOCX_NS):
        rid = rel.get("Id")
        target = rel.get("Target")
        if rid and target:
            rid_map[rid] = target
    return rid_map


def iter_paragraphs(docx_path: str):
    """Yield (text, element) for each paragraph in document.xml."""
    doc_path = "word/document.xml"
    with zipfile.ZipFile(docx_path, "r") as zf:
        if doc_path not in zf.namelist():
            return
        data = zf.read(doc_path)
    root = ET.fromstring(data)
    for p in root.iter(f"{{{DOCX_NS['w']}}}p"):
        texts = []
        for t in p.iter(f"{{{DOCX_NS['w']}}}t"):
            if t.text:
                texts.append(t.text)
        yield "".join(texts), p


def find_image_references(docx_path: str) -> list[dict]:
    """Find paragraphs that reference images via blip/embed relationships."""
    refs: list[dict] = []
    for para_text, p in iter_paragraphs(docx_path):
        blips = p.findall(".//a:blip", DOCX_NS)
        for blip in blips:
            embed = blip.get(f"{{{DOCX_NS['r']}}}embed")
            link = blip.get(f"{{{DOCX_NS['r']}}}link")
            refs.append({
                "paragraph_text": para_text,
                "rId_embed": embed,
                "rId_link": link,
            })
    return refs


def guess_caption(paragraph_text: str) -> str | None:
    """Heuristic: detect Figure/图/Table/表 captions."""
    if not paragraph_text:
        return None
    patterns = [
        r"^(?:Figure|Fig\.?|图)\s*\d+[.:、\s]*(.+)",
        r"^(?:Table|Tab\.?|表)\s*\d+[.:、\s]*(.+)",
    ]
    for pat in patterns:
        m = re.match(pat, paragraph_text.strip(), re.IGNORECASE)
        if m:
            return paragraph_text.strip()
    return None


def build_index(docx_path: str, media_items: list[dict], rid_map: dict[str, str]) -> list[dict]:
    """Combine media list with paragraph context and captions."""
    refs = find_image_references(docx_path)
    index: list[dict] = []
    for idx, item in enumerate(media_items, start=1):
        target_name = Path(item["archive_path"]).name
        matching_refs = [
            r for r in refs
            if (r["rId_embed"] and rid_map.get(r["rId_embed"], "").endswith(target_name))
            or (r["rId_link"] and rid_map.get(r["rId_link"], "").endswith(target_name))
        ]
        para_text = matching_refs[0]["paragraph_text"] if matching_refs else ""
        caption = guess_caption(para_text)
        index.append({
            "index": idx,
            "filename": item["filename"],
            "extension": item["extension"],
            "size_bytes": item["size_bytes"],
            "saved_path": item["saved_path"],
            "paragraph_context": para_text[:500],
            "caption": caption,
            "has_text_content": caption is not None or len(para_text) > 0,
        })
    return index


def write_summary(out_dir: Path, index: list[dict]) -> None:
    summary_path = out_dir / "summary.txt"
    total = len(index)
    with_text = sum(1 for x in index if x["has_text_content"])
    with_caption = sum(1 for x in index if x["caption"])
    by_ext: dict[str, int] = {}
    for x in index:
        by_ext[x["extension"]] = by_ext.get(x["extension"], 0) + 1

    lines = [
        f"Total embedded media: {total}",
        f"Items with paragraph context: {with_text}",
        f"Items with detected caption: {with_caption}",
        "By extension:",
    ]
    for ext, count in sorted(by_ext.items()):
        lines.append(f"  {ext or '(none)'}: {count}")
    if with_caption:
        lines.append("\nDetected captions:")
        for x in index:
            if x["caption"]:
                lines.append(f"  [{x['index']}] {x['caption']}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    docx_path = Path(args.docx).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not docx_path.exists():
        raise FileNotFoundError(docx_path)

    media_dir = out_dir / "media"
    ensure_dir(media_dir)

    media_items = copy_media(str(docx_path), media_dir)
    rid_map = build_rid_to_target(str(docx_path))
    index = build_index(str(docx_path), media_items, rid_map)

    with open(out_dir / "media_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    write_summary(out_dir, index)

    print(f"Extracted {len(index)} media item(s) to {media_dir}")
    print(f"Index: {out_dir / 'media_index.json'}")
    print(f"Summary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
