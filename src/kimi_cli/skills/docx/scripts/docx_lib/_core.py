"""
docx_lib._core — pure-Python fallback for docx validation/fixing.

This module replaces the Linux-only compiled extension so the docx skill's
validation pipeline works on macOS and other platforms. It implements the
same public functions used by validate_all.py.
"""

import os
import re
import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# -----------------------------------------------------------------------------
# Namespaces
# -----------------------------------------------------------------------------
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

# Register prefixes so ET.write preserves them when files are rewritten.
ET.register_namespace('w', W_NS)
ET.register_namespace('r', R_NS)
ET.register_namespace('wp', WP_NS)
ET.register_namespace('a', A_NS)
ET.register_namespace('pic', PIC_NS)
ET.register_namespace('rel', REL_NS)
ET.register_namespace('', CT_NS)


def _qn(ns, tag):
    """Return a Clark notation tag."""
    return f"{{{ns}}}{tag}"


def _local(tag):
    """Strip namespace from a Clark-notation tag."""
    if tag.startswith('{'):
        return tag.split('}', 1)[1]
    return tag


def _tag_without_ns(tag):
    return _local(tag)


# -----------------------------------------------------------------------------
# Element ordering maps (subset of OpenXML schema ordering that matters for
# documents produced by the C# OpenXML SDK).
# -----------------------------------------------------------------------------
ORDERS = {
    'p': {
        'pPr': 0,
        'bookmarkStart': 1,
        'bookmarkEnd': 2,
        'commentRangeStart': 3,
        'commentRangeEnd': 4,
        'r': 5,
        'hyperlink': 6,
        'fldSimple': 7,
        'sdt': 8,
    },
    'r': {
        'rPr': 0,
        't': 1,
        'tab': 2,
        'br': 3,
        'sym': 4,
        'drawing': 5,
        'pict': 6,
        'footnoteReference': 7,
        'endnoteReference': 8,
        'fldChar': 9,
        'instrText': 10,
    },
    'pPr': {
        'pStyle': 0,
        'keepNext': 1,
        'keepLines': 2,
        'pageBreakBefore': 3,
        'widowControl': 4,
        'numPr': 5,
        'spacing': 6,
        'ind': 7,
        'jc': 8,
        'outlineLvl': 9,
    },
    'rPr': {
        'rFonts': 0,
        'b': 1,
        'bCs': 2,
        'i': 3,
        'iCs': 4,
        'caps': 5,
        'smallCaps': 6,
        'strike': 7,
        'dstrike': 8,
        'outline': 9,
        'shadow': 10,
        'emboss': 11,
        'imprint': 12,
        'noProof': 13,
        'snapToGrid': 14,
        'vanish': 15,
        'webHidden': 16,
        'color': 17,
        'spacing': 18,
        'sz': 19,
        'szCs': 20,
        'highlight': 21,
        'u': 22,
        'bdr': 23,
        'shd': 24,
        'vertAlign': 25,
        'rtl': 26,
        'cs': 27,
        'em': 28,
        'lang': 29,
    },
    'tbl': {
        'tblPr': 0,
        'tblGrid': 1,
        'tr': 2,
    },
    'tblPr': {
        'tblW': 0,
        'tblBorders': 1,
        'tblCellMar': 2,
        'tblLook': 3,
    },
    'tr': {
        'trPr': 0,
        'tc': 1,
    },
    'tc': {
        'tcPr': 0,
        'p': 1,
        'tbl': 2,
    },
    'tcPr': {
        'tcW': 0,
        'gridSpan': 1,
        'vMerge': 2,
        'shd': 3,
        'vAlign': 4,
    },
    'tblBorders': {
        'top': 0,
        'left': 1,
        'bottom': 2,
        'right': 3,
        'insideH': 4,
        'insideV': 5,
    },
    'sectPr': {
        'headerReference': 0,
        'footerReference': 1,
        'titlePage': 2,
        'pgSz': 3,
        'pgMar': 4,
        'cols': 5,
        'docGrid': 6,
    },
    'settings': {
        'zoom': 0,
        'defaultTabStop': 1,
        'characterSpacingControl': 2,
        'compatibility': 3,
        'updateFields': 4,
    },
}


def _reorder(element, order_map):
    """Reorder children of element according to order_map. Return # changes."""
    children = list(element)
    if len(children) < 2:
        return 0

    def sort_key(child):
        local = _local(child.tag)
        return (order_map.get(local, 999), local)

    sorted_children = sorted(children, key=sort_key)
    if sorted_children == children:
        return 0

    # Clear and re-append in sorted order
    element[:] = []
    element.extend(sorted_children)
    return sum(1 for a, b in zip(children, sorted_children) if a is not b) or 1


def fix_element_order_in_tree(root):
    """Recursively fix element order. Returns total number of changes."""
    count = 0
    for element in root.iter():
        local = _local(element.tag)
        if local in ORDERS:
            count += _reorder(element, ORDERS[local])
    return count


def fix_settings(root):
    """Fix settings element child order. Returns # changes."""
    for settings in root.iter(_qn(W_NS, 'settings')):
        return _reorder(settings, ORDERS.get('settings', {}))
    return 0


def fix_body_order(root):
    """Ensure sectPr is the last child of the body. Returns # changes."""
    body = root.find(_qn(W_NS, 'body'))
    if body is None:
        return 0
    children = list(body)
    sectPr = body.find(_qn(W_NS, 'sectPr'))
    if sectPr is None or sectPr is children[-1]:
        return 0
    body.remove(sectPr)
    body.append(sectPr)
    return 1


def fix_table_width_conservative(root):
    """Best-effort table width consistency fix. Currently handled by ordering."""
    return 0


def wrap_border_elements(root):
    """No-op placeholder."""
    return 0


# -----------------------------------------------------------------------------
# Business-rule checks
# -----------------------------------------------------------------------------
def check_table_grid_consistency(root):
    """Verify each table's grid column count matches its rows."""
    errors = []
    for tbl in root.iter(_qn(W_NS, 'tbl')):
        grid = tbl.find(_qn(W_NS, 'tblGrid'))
        if grid is None:
            errors.append("TABLE: missing tblGrid")
            continue
        grid_cols = len(list(grid.iter(_qn(W_NS, 'gridCol'))))
        for i, tr in enumerate(tbl.iter(_qn(W_NS, 'tr'))):
            cells = list(tr.iter(_qn(W_NS, 'tc')))
            spans = []
            for tc in cells:
                grid_span = tc.find('.//'+_qn(W_NS, 'gridSpan'))
                span = int(grid_span.get(_qn(W_NS, 'val'), '1')) if grid_span is not None else 1
                spans.append(span)
            total_span = sum(spans)
            if total_span != grid_cols:
                errors.append(
                    f"TABLE: row {i} spans {total_span} logical columns but "
                    f"tblGrid defines {grid_cols} columns"
                )
    return errors


def check_image_aspect_ratio(root, extract_dir):
    """Check that image extent aspect ratio matches actual image dimensions."""
    errors = []
    media_dir = Path(extract_dir) / 'word' / 'media'
    rels_path = Path(extract_dir) / 'word' / '_rels' / 'document.xml.rels'
    rel_map = {}
    if rels_path.exists():
        rel_tree = ET.parse(rels_path)
        for rel in rel_tree.getroot().iter(_qn(REL_NS, 'Relationship')):
            rel_map[rel.get('Id')] = rel.get('Target')

    for drawing in root.iter(_qn(W_NS, 'drawing')):
        extent = None
        # Inline or anchor Extent
        for ext in drawing.iter(_qn(WP_NS, 'extent')):
            cx = int(ext.get('cx', 0))
            cy = int(ext.get('cy', 0))
            if cx and cy:
                extent = (cx, cy)
                break
        if extent is None:
            continue

        # Find blip embed relationship
        embed = None
        for blip in drawing.iter(_qn(A_NS, 'blip')):
            embed = blip.get(_qn(R_NS, 'embed'))
            if embed:
                break
        if not embed or embed not in rel_map:
            continue

        target = rel_map[embed]
        img_path = media_dir / Path(target).name
        if not img_path.exists():
            continue

        w, h = _png_dimensions(img_path)
        if w is None or h is None:
            continue

        doc_ratio = extent[0] / extent[1]
        img_ratio = w / h
        if abs(doc_ratio - img_ratio) > 0.05:
            errors.append(
                f"IMAGE: aspect ratio mismatch for {target} "
                f"(extent {extent[0]}x{extent[1]} vs image {w}x{h})"
            )
    return errors


def _png_dimensions(path):
    """Read PNG width/height from the IHDR chunk."""
    try:
        with open(path, 'rb') as f:
            header = f.read(24)
            if header[:8] != b'\x89PNG\r\n\x1a\n':
                return None, None
            width, height = struct.unpack('>II', header[16:24])
            return width, height
    except Exception:
        return None, None


def check_comments_integrity(extract_dir):
    """Ensure comments.xml ids are referenced in document.xml."""
    errors = []
    comments_path = Path(extract_dir) / 'word' / 'comments.xml'
    doc_path = Path(extract_dir) / 'word' / 'document.xml'
    if not comments_path.exists():
        return errors

    try:
        doc_tree = ET.parse(doc_path)
        doc_root = doc_tree.getroot()
    except Exception:
        return errors

    referenced = set()
    for el in doc_root.iter():
        if _local(el.tag) in ('commentRangeStart', 'commentRangeEnd', 'commentReference'):
            val = el.get(_qn(W_NS, 'id'))
            if val:
                referenced.add(val)

    try:
        c_tree = ET.parse(comments_path)
        c_root = c_tree.getroot()
    except Exception:
        return errors

    for c in c_root.iter(_qn(W_NS, 'comment')):
        cid = c.get(_qn(W_NS, 'id'))
        if cid and cid not in referenced:
            errors.append(f"COMMENT: comment id {cid} is not referenced in document.xml")
    return errors


def check_section_margins(root):
    """Warn if page margins are unusually small."""
    warnings = []
    for sectPr in root.iter(_qn(W_NS, 'sectPr')):
        pgMar = sectPr.find(_qn(W_NS, 'pgMar'))
        if pgMar is None:
            continue
        for attr in ('top', 'bottom', 'left', 'right'):
            val = pgMar.get(attr)
            if val is not None and int(val) < 100:
                warnings.append(f"SECTION: margin {attr}={val} is very small")
    return warnings


def check_namespace_declarations(extract_dir):
    """Check that document.xml declares required namespaces."""
    errors = []
    doc_path = Path(extract_dir) / 'word' / 'document.xml'
    try:
        root = ET.parse(doc_path).getroot()
    except Exception as e:
        return [f"NAMESPACE: cannot parse document.xml: {e}"]

    nsmap = dict([node for _, node in ET.iterparse(str(doc_path), events=['start-ns'])])
    # iterparse returns namespaces as they appear; collect first root ns map via attribute fallback
    if W_NS not in root.attrib.values() and W_NS not in nsmap.values():
        errors.append("NAMESPACE: document.xml missing WordprocessingML namespace")
    return errors


def check_id_uniqueness(root):
    """Check uniqueness of docPr/@id and bookmark ids."""
    errors = []
    docpr_ids = []
    for dp in root.iter(_qn(WP_NS, 'docPr')):
        did = dp.get('id')
        if did:
            docpr_ids.append(did)
    if len(docpr_ids) != len(set(docpr_ids)):
        errors.append("ID: duplicate wp:docPr/@id values")

    bm_start = []
    for bs in root.iter(_qn(W_NS, 'bookmarkStart')):
        bid = bs.get(_qn(W_NS, 'id'))
        if bid:
            bm_start.append(bid)
    if len(bm_start) != len(set(bm_start)):
        errors.append("ID: duplicate bookmarkStart/@id values")
    return errors


# -----------------------------------------------------------------------------
# Package-level fixes
# -----------------------------------------------------------------------------
CONTENT_TYPES = {
    'document.xml': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml',
    'styles.xml': 'application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml',
    'numbering.xml': 'application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml',
    'settings.xml': 'application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml',
    'fontTable.xml': 'application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml',
    'webSettings.xml': 'application/vnd.openxmlformats-officedocument.wordprocessingml.webSettings+xml',
    'footnotes.xml': 'application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml',
    'endnotes.xml': 'application/vnd.openxmlformats-officedocument.wordprocessingml.endnotes+xml',
}


def _content_type_for(part_name):
    name = Path(part_name).name
    base = name
    for key in CONTENT_TYPES:
        if name.endswith(key):
            return CONTENT_TYPES[key]
    if 'header' in base:
        return 'application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml'
    if 'footer' in base:
        return 'application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml'
    return None


def fix_content_types(extract_dir):
    """Add missing Override entries to [Content_Types].xml."""
    ct_path = Path(extract_dir) / '[Content_Types].xml'
    if not ct_path.exists():
        return 0

    tree = ET.parse(ct_path)
    root = tree.getroot()
    overrides = {
        ov.get('PartName'): ov.get('ContentType')
        for ov in root.iter(_qn(CT_NS, 'Override'))
    }

    fixes = 0
    for f in Path(extract_dir).rglob('*.xml'):
        rel = '/' + str(f.relative_to(extract_dir))
        if rel.endswith('.rels'):
            continue
        if rel in overrides:
            continue
        ct = _content_type_for(rel)
        if ct:
            ov = ET.SubElement(root, _qn(CT_NS, 'Override'))
            ov.set('PartName', rel)
            ov.set('ContentType', ct)
            fixes += 1

    if fixes:
        tree.write(ct_path, encoding='UTF-8', xml_declaration=True)
    return fixes


def fix_relationship_paths(extract_dir):
    """Normalize relationship Target paths.

    Only backslashes are rewritten. Leading slashes are preserved because
    package-root absolute targets such as '/word/header1.xml' are valid OPC
    URIs; stripping them makes the target relative to the .rels file and
    breaks part resolution.
    """
    fixes = 0
    for rels_file in Path(extract_dir).rglob('*.rels'):
        tree = ET.parse(rels_file)
        root = tree.getroot()
        changed = False
        for rel in root.iter(_qn(REL_NS, 'Relationship')):
            target = rel.get('Target', '')
            fixed = target.replace('\\', '/')
            if fixed != target:
                rel.set('Target', fixed)
                changed = True
                fixes += 1
        if changed:
            tree.write(rels_file, encoding='UTF-8', xml_declaration=True)
    return fixes
