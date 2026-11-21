"""Build the arXiv source package for this paper.

Flattens paper.tex (shared preamble and sections inlined), switches the
bibliography from biblatex to natbib/plainnat so the shipped .bbl does not depend
on a biblatex version, copies the vector figures, keeps only the cited
references, compiles once with Tectonic to produce main.bbl and to check the
package is self-contained, and writes thurstone_portfolios_arxiv.zip.

Run from anywhere:  python3 papers/thurstone-portfolios/arxiv/make_arxiv.py
"""
import os, re, shutil, subprocess, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
PAPERS = os.path.dirname(PAPER)
BUILD = os.path.join(HERE, "build")
ZIP = os.path.join(HERE, "thurstone_portfolios_arxiv.zip")


def read(p):
    with open(p) as f:
        return f.read()


def main():
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(os.path.join(BUILD, "figures"))

    tex = read(os.path.join(PAPER, "paper.tex"))
    pre = read(os.path.join(PAPERS, "shared", "preamble.tex"))
    # natbib in place of biblatex
    pre = re.sub(r"\\usepackage\[backend=bibtex[^\]]*\]\{biblatex\}\n", r"\\usepackage[round]{natbib}\n", pre)
    pre = re.sub(r"\\addbibresource\{[^}]*\}\n", "", pre)
    # drop the comment block that explains the biblatex choice
    pre = "\n".join(l for l in pre.splitlines()
                    if not (l.startswith("%") and re.search(r"bib|Tectonic|biber|mismatch|input|Shared preamble|common to all papers", l))) + "\n"
    assert "biblatex}" not in pre and "natbib" in pre
    tex = tex.replace("\\input{../shared/preamble.tex}", pre)
    tex = tex.replace("\\appto\\bibsetup{\\sloppy}\n", "")
    tex = re.sub(r"\\input\{sections/([^}]+)\}",
                 lambda m: read(os.path.join(PAPER, "sections", m.group(1) + ".tex")), tex)
    tex = tex.replace("\\printbibliography",
                      "\\begingroup\\sloppy\n\\bibliographystyle{plainnat}\n\\bibliography{refs}\n\\endgroup")
    tex = tex.replace("\\parencite", "\\citep").replace("\\textcite", "\\citet")
    assert "\\input{" not in tex and "{biblatex}" not in tex
    with open(os.path.join(BUILD, "main.tex"), "w") as f:
        f.write(tex)

    # figures actually included
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        shutil.copy(os.path.join(PAPER, fig), os.path.join(BUILD, fig))

    # cited references only
    keys = set()
    for group in re.findall(r"\\cite[pt]\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        keys.update(k.strip() for k in group.split(","))
    bib = read(os.path.join(PAPERS, "refs.bib"))
    entries = re.split(r"\n(?=@)", bib)
    kept = [e for e in entries if (m := re.match(r"@\w+\{([^,]+),", e)) and m.group(1) in keys]
    found = {re.match(r"@\w+\{([^,]+),", e).group(1) for e in kept}
    assert found == keys, f"missing from refs.bib: {sorted(keys - found)}"
    with open(os.path.join(BUILD, "refs.bib"), "w") as f:
        f.write("\n\n".join(e.strip() for e in kept) + "\n")

    subprocess.run(["tectonic", "--keep-intermediates", "--keep-logs", "main.tex"], cwd=BUILD, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = read(os.path.join(BUILD, "main.log"))
    bad = [l for l in log.splitlines() if "undefined" in l.lower() or l.startswith("Overfull")]
    assert not bad, bad[:5]
    assert os.path.exists(os.path.join(BUILD, "main.bbl"))

    members = ["main.tex", "main.bbl", "refs.bib"] + sorted(
        os.path.join("figures", f) for f in os.listdir(os.path.join(BUILD, "figures")))
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for m in members:
            z.write(os.path.join(BUILD, m), m)
    shutil.copy(os.path.join(BUILD, "main.pdf"), os.path.join(HERE, "arxiv_preview.pdf"))
    print("wrote", ZIP)
    for m in members:
        print("  ", m)
    print(f"{len(keys)} references; preview at arxiv/arxiv_preview.pdf")


if __name__ == "__main__":
    main()
