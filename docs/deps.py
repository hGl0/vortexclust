import re, sys, itertools

IN, OUT = sys.argv[1], sys.argv[2]

PKG_PREFIX = "vortexclust."
SUBMODULES = ["analysis", "core", "io", "models", "workflows", "visualization"]

# ---- Palette (fill, border) per submodule ----
SUBMODULE_COLORS = {
    "analysis":       ("#E6EEF9", "#345995"),
    "visualization":  ("#E9F5EC", "#2E7D32"),
    "core":           ("#EEF0F7", "#364F6B"),
    "models":         ("#FFF0F2", "#B23A48"),
    "io":             ("#F3E8FF", "#7E5EB8"),
    "workflows":      ("#FFF6EA", "#C07C2E"),
}
FALLBACK_FILLS   = ["#FFF0F2","#EEF0F7","#FFF7E6","#E7FAF2","#F0ECFF","#F6F9FF"]
FALLBACK_BORDERS = ["#B23A48","#364F6B","#C07C2E","#2E7D32","#7E5EB8","#345995"]

dot = open(IN, "r", encoding="utf-8").read()

# Ensure digraph (pyreverse sometimes writes 'graph')
if not re.search(r'^\s*digraph\b', dot, flags=re.M):
    dot = re.sub(r'^\s*graph\b', 'digraph', dot, flags=re.M)

# --- collect nodes & shorten labels ---
node_pat = re.compile(r'(^\s*"([^"]+)"\s*)(\[(.*?)\])?\s*;\s*$', re.M | re.S)

def leaf(name: str) -> str:
    return name.split(".")[-1] if name.startswith(PKG_PREFIX) else name

def scrub_label(a: str) -> str:
    a = re.sub(r'(?<![\w])label\s*=\s*("[^"]*"|[^,\]]+)\s*,?', '', a)
    a = re.sub(r',\s*,', ',', a).strip().strip(',')
    return a

# rewrite nodes with short labels
dot = node_pat.sub(lambda m: (
    f'{m.group(1)}[{scrub_label(m.group(4) or "") + (", " if (m.group(4) or "").strip() else "")}label="{leaf(m.group(2))}"];'
    if m.group(2).startswith(PKG_PREFIX) else m.group(0)
), dot)

# --- group by submodule; include parent nodes too ---
groups = {s: [] for s in SUBMODULES}
for m in node_pat.finditer(dot):
    nid = m.group(2)
    if not nid.startswith(PKG_PREFIX):
        continue
    parts = nid.split(".")
    if len(parts) >= 2:
        sub = parts[1]
        if sub in groups:
            groups[sub].append(nid)

# --- build color clusters only (no layout constraints) ---
palette = itertools.cycle(zip(FALLBACK_FILLS, FALLBACK_BORDERS))
cluster_blocks = []
for sub in SUBMODULES:
    fill, border = SUBMODULE_COLORS.get(sub, next(palette))
    cid = f'cluster_{sub}'
    block = [
        f'subgraph {cid} {{',
        f'  label="{sub}";',
        f'  labelloc=b; labeljust=c;',
        f'  style="filled";',
        f'  color="{border}";',
        f'  fillcolor="{fill}";',
        f'  graph [nodesep=0.10, ranksep=0.1, margin="0.1,0.1"];',
        f'  node  [style=filled, fillcolor="{fill}"];',
    ]
    if sub in (["visualization"]):
        block = [
            f'subgraph {cid} {{',
            f'  label="{sub}";',
            f'  labelloc=t; labeljust=c;',
            f'  style="filled";',
            f'  color="{border}";',
            f'  fillcolor="{fill}";',
            f'  graph [nodesep=0.10, ranksep=0.1, margin="0.1,0.1"];',
            f'  node  [style=filled, fillcolor="{fill}"];',
        ]
    if groups[sub]:
        for nid in sorted(groups[sub]):
            block.append(f'  "{nid}";')
    else:
        # keep an empty module visible if it has no nodes
        block.append(f'  "__DUMMY_{sub}" [shape=point, style=invis, width=0.01, height=0.01, label=""];')
    block.append('}')
    cluster_blocks.append("\n".join(block))

# --- append clusters before final brace ---
dot = re.sub(r'(}\s*)\Z', "\n// === submodule clusters ===\n" + "\n".join(cluster_blocks) + r"\n}\n", dot, flags=re.S)

# --- CUSTOM FOOTER: root graph attrs + positioning edges (written LAST) --
# 1) Root graph attributes (last-wins)
EXTRA_GRAPH_ATTRS = {
    "splines": "ortho",
    "concentrate": "true",
    "compound": "true",
    "ranksep" : 0.5
}

# 2) Positioning edges you want to force (add as many as you like)
#    Each item: (src, dst, {attr: value, ...})
POSITION_EDGES = [
    ("vortexclust", "vortexclust.visualization", {"style": "invis", "weight": 10}),
    ("vortexclust", "vortexclust.workflows.extended_demo_script", {"style": "invis", "weight": 10}),
    ("vortexclust.visualization.utils", "vortexclust.analysis.aggregator", {"style": "invis", "weight": 10}),
    # ("vortexclust.core.utils", "vortexclust.models.validation", {"style": "invis", "weight": 10}),
    ("vortexclust.workflows", "vortexclust.analysis", {"style": "invis", "weight": 10}),
    ("vortexclust.analysis.sampling", "vortexclust.models.validation", {"style" :"invis", "weight":10})
]

def _fmt_attrs(d):
    parts = []
    for k, v in d.items():
        # quote only if value has spaces or punctuation typical of strings
        if isinstance(v, str) and any(c in v for c in ' ,:;"'):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f'{k}={v}')
    return ", ".join(parts)

footer_lines = []
if EXTRA_GRAPH_ATTRS:
    footer_lines.append(f'  graph [{_fmt_attrs(EXTRA_GRAPH_ATTRS)}];')
for s, t, attrs in POSITION_EDGES:
    attrs_str = _fmt_attrs(attrs) if attrs else ""
    footer_lines.append(f'  "{s}" -> "{t}"[{(" " + attrs_str) if attrs_str else ""}];')

# Write the footer just before the final closing brace
dot = re.sub(r'(}\s*)\Z', "\n" + "\n".join(footer_lines) + r"\n}\n", dot, flags=re.S)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(dot)

print(f"Compile with 'dot -Tpng {OUT} -o out.png'")
