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

ALGO = r"""<div id="alg:tilt" class="algorithm">
<p class="algo-title"><strong>Algorithm 1.</strong> Ability tilt (Thurstone portfolio)</p>
<ol class="algo">
<li><span class="kw">require</span> target weights \(w^{\mathrm{target}}\in\Delta\); reference sampler \(S_{\mathrm{calib}}\); target sampler \(S_{\mathrm{tilt}}\) (Gaussian \(\mathcal{N}(\cdot,\,C_{\mathrm{tilt}})\) by default).</li>
<li>\(a \gets \operatorname{Calibrate}(w^{\mathrm{target}}, S_{\mathrm{calib}})\) <span class="cmt">&#9655; abilities s.t. the race under \(S_{\mathrm{calib}}\) yields \(w^{\mathrm{target}}\)</span></li>
<li>draw \(X^{(1)},\dots,X^{(M)} \sim S_{\mathrm{tilt}}(a)\) from <em>fixed</em> seeds <span class="cmt">&#9655; Gaussian: \(X^{(m)} = a + C_{\mathrm{tilt}}^{1/2} Z^{(m)}\)</span></li>
<li>\(w_i \gets \tfrac{1}{M}\sum_{m=1}^{M} \mathbf{1}\!\left[\, i = \arg\min_k X^{(m)}_k \,\right]\) <span class="cmt">&#9655; win frequency</span></li>
<li><span class="kw">return</span> \(w\)</li>
</ol>
</div>"""

TYPE = {"sec": "Section", "prop": "Proposition", "thm": "Theorem", "lem": "Lemma",
        "cor": "Corollary", "def": "Definition", "rem": "Section", "fig": "Figure",
        "tab": "Table", "alg": "Algorithm", "eq": "Equation"}
LABEL = r"(?:sec|prop|thm|lem|cor|def|rem|fig|tab|alg|eq):[^\],]+"
WORDS = "|".join(sorted(set(TYPE.values())))
# already introduced: "...Proposition&nbsp;" or "...Propositions&nbsp;1 and " (the tail of a multi-ref list)
INTRO = re.compile(rf"(?:{WORDS})s?[\s\u00a0](?:[\d.]+(?:,\s*|\s+and\s+))*$")

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

num.setdefault("alg:tilt", "1")      # the hand-built ALGO block above is Algorithm 1


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

# 3) SINGLE \Cref leaks. pandoc renders these as a bare number ("the turnover bound of 2"), and an
#    unresolved one as a raw "[alg:tilt]". Prefix each with its type word unless the sentence already
#    introduced it -- including the tail of a multi-ref list such as "Propositions 1 and <5>".
def plain_tail(s, n=70):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s[-400:]))[-n:]

def name_single(m):
    lab = m.group("lab")
    word = TYPE.get(lab.split(":")[0], "Section")
    if INTRO.search(plain_tail(html[:m.start()])):
        return m.group(0)                                  # already named in the running text
    return f'{word}&nbsp;{m.group(0)}'

# 0) an anchor whose entire text is a raw bracketed label carries no number; drop the wrapper and let the
#    raw-label expansion below build it properly, so we never nest one anchor inside another.
html, nuw = re.subn(rf'<a\b[^>]*>\[({LABEL})\]</a>', r"[\1]", html)

# equations first: their numbers are assigned by MathJax at RENDER time, so the HTML cannot know them and
# pandoc leaves a "?" behind. Hand them to MathJax as \eqref, which resolves to "(6)" and links to the display.
EQ = r"eq:[^\],\"]+"
html, ne = re.subn(rf'(?:Equation&nbsp;)?<a href="#({EQ})"[^>]*>[^<]{{0,12}}</a>',
                   lambda m: rf'\(\eqref{{{m.group(1)}}}\)', html)
html, ne2 = re.subn(rf"\[({EQ})\]", lambda m: rf'\(\eqref{{{m.group(1)}}}\)', html)

# an unresolved raw label, e.g. "is [alg:tilt]", never became a link at all
def name_raw(m):
    lab = m.group(1)
    word = TYPE.get(lab.split(":")[0], "Section")
    return f'{word}&nbsp;<a href="#{lab}">{num.get(lab, "?")}</a>'
html, n4 = re.subn(rf"\[({LABEL})\]", name_raw, html)

single = re.compile(rf'<a href="#(?P<lab>{LABEL})"[^>]*>[^<]{{1,12}}</a>')
html, n3 = single.subn(name_single, html)

# pandoc has no reader for this class's \paragraph, so a run-in heading arrives as
#   <p><span>3.25ex </span><span>-1em</span><span><em></em></span><span>Title.</span> body...
# i.e. the \vspace/\hspace dimensions leak as body text. Restore it as a run-in heading.
RUNIN = re.compile(r'<p><span>[-\d.]+ex\s*</span>\s*<span>[-\d.]+em</span>\s*'
                   r'<span><em></em></span>\s*<span>(.*?)</span>', re.S)
html, nr = re.subn(RUNIN, lambda m: f'<p><strong class="runin">{m.group(1)}</strong>', html)

open(path, "w", encoding="utf-8").write(html)
print(f"  postprocess: algorithm + {nfix + n2} multi-cref + {n3} named single-cref + {n4} raw label"
      f" + {ne + ne2} eqref + {nr} run-in + {nuw} unwrapped")
