#!/usr/bin/env python3
"""Post-process the pandoc HTML render of a paper.

Fixes two things pandoc gets wrong for this LaTeX:
  1. the algpseudocode `algorithm` environment (no pandoc reader) collapses into a
     run-on and leaks \\textsc{...} through MathJax;
  2. multi-label \\Cref{a,b} cross-references leak as a single link whose text is
     the raw "[a,b]" label list (single \\Cref works; comma-lists do not).
The single LaTeX source stays authoritative for the PDF; we repair only the web.
"""
import re
import sys

ALGO = r"""<div class="algorithm">
<p class="algo-title"><strong>Algorithm 1.</strong> Ability tilt (Thurstone portfolio)</p>
<ol class="algo">
<li><span class="kw">require</span> target weights \(w^{\mathrm{target}}\in\Delta\); reference sampler \(S_{\mathrm{calib}}\); target sampler \(S_{\mathrm{tilt}}\) (Gaussian \(\mathcal{N}(\cdot,\,C_{\mathrm{tilt}})\) by default).</li>
<li>\(a \gets \operatorname{Calibrate}(w^{\mathrm{target}}, S_{\mathrm{calib}})\) <span class="cmt">&#9655; abilities s.t.\ the race under \(S_{\mathrm{calib}}\) yields \(w^{\mathrm{target}}\)</span></li>
<li>draw \(X^{(1)},\dots,X^{(M)} \sim S_{\mathrm{tilt}}(a)\) from <em>fixed</em> seeds <span class="cmt">&#9655; Gaussian: \(X^{(m)} = a + C_{\mathrm{tilt}}^{1/2} Z^{(m)}\)</span></li>
<li>\(w_i \gets \tfrac{1}{M}\sum_{m=1}^{M} \mathbf{1}\!\left[\, i = \arg\min_k X^{(m)}_k \,\right]\) <span class="cmt">&#9655; win frequency</span></li>
<li><span class="kw">return</span> \(w\)</li>
</ol>
</div>"""

TYPE = {"sec": "Section", "prop": "Proposition", "thm": "Theorem",
        "rem": "Section", "fig": "Figure", "eq": "Equation"}
LABEL = r"(?:sec|prop|thm|rem|fig|eq):[^\],]+"

path = sys.argv[1]
html = open(path, encoding="utf-8").read()

# 1) algorithm
html, _ = re.subn(r'<div class="algorithm">.*?</div>\s*</div>', lambda _m: ALGO,
                  html, count=1, flags=re.S)

# 2) label -> number map: numbered headings (--number-sections) and clean xrefs.
num = {}
for tag in re.findall(r"<h[1-6][^>]*>", html):
    mid = re.search(r'id="([^"]+)"', tag)
    mno = re.search(r'data-number="([^"]+)"', tag)
    if mid and mno:
        num[mid.group(1)] = mno.group(1)
for lab, txt in re.findall(r'data-reference="([^"]+)"[^>]*>([^<]+)</a>', html):
    if "[" not in txt:
        num.setdefault(lab, txt.strip())

def link(lab):
    return f'<a href="#{lab}">{num.get(lab, "?")}</a>'

def render(m):
    labels = [s.strip() for s in m.group(1).split(",")]
    types = [TYPE.get(l.split(":")[0], "Section") for l in labels]
    links = [link(l) for l in labels]
    if len(set(types)) == 1:                       # "Sections 6 and 7"
        word = types[0] + "s"
        joined = (" and ".join(links) if len(links) == 2
                  else ", ".join(links[:-1]) + ", and " + links[-1])
        return f"{word}&nbsp;{joined}"
    chunks = [f"{t}&nbsp;{l}" for t, l in zip(types, links)]   # mixed types
    return (" and ".join(chunks) if len(chunks) == 2
            else ", ".join(chunks[:-1]) + ", and " + chunks[-1])

# the leak is an <a> whose text is the raw bracketed list; also catch bare lists
pat = rf"<a\b[^>]*>\[({LABEL}(?:,{LABEL})+)\]</a>"
html, nfix = re.subn(pat, render, html)
html, n2 = re.subn(rf"\[({LABEL}(?:,{LABEL})+)\]", render, html)

open(path, "w", encoding="utf-8").write(html)
print(f"  postprocess: algorithm + {nfix + n2} multi-cref(s)")
