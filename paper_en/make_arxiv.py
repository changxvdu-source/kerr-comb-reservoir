"""Build the arXiv upload from the English paper: arxiv/ and arxiv_submission.tar.gz in the project root.

arXiv compiles the sources itself, so the package holds main.tex with the figure path changed to figures/, the sections, the
generated tables, the figures the text includes (PDF), and main.bbl from a finished local build (arXiv does not run BibTeX on
uploaded .bib files reliably; the .bbl must match the .tex). Run it after compiling paper_en, from the project root:

    python paper_en/make_arxiv.py
"""
import glob
import os
import re
import shutil
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper_en")
OUT = os.path.join(ROOT, "arxiv")


def main():
    bbl = os.path.join(SRC, "main.bbl")
    if not os.path.exists(bbl):
        raise SystemExit("paper_en/main.bbl is missing: compile paper_en first (pdflatex, bibtex, pdflatex, pdflatex)")
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "sections"))
    os.makedirs(os.path.join(OUT, "tables"))
    os.makedirs(os.path.join(OUT, "figures"))
    main_tex = open(os.path.join(SRC, "main.tex"), encoding="utf-8").read()
    main_tex = main_tex.replace(r"\graphicspath{{../figures/}}", r"\graphicspath{{figures/}}")
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(main_tex)
    texts = [main_tex]
    for sub in ("sections", "tables"):
        for f in sorted(glob.glob(os.path.join(SRC, sub, "*.tex"))):
            shutil.copy(f, os.path.join(OUT, sub))
            texts.append(open(f, encoding="utf-8").read())
    shutil.copy(bbl, OUT)
    used = set()
    for t in texts:
        used |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t))
    for name in sorted(used):
        shutil.copy(os.path.join(ROOT, "figures", name + ".pdf"), os.path.join(OUT, "figures"))
    todo = [t for t in texts if r"\todo{" in t.replace(r"\newcommand{\todo}", "")]
    archive = os.path.join(ROOT, "arxiv_submission.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(glob.glob(os.path.join(OUT, "**", "*"), recursive=True)):
            if os.path.isfile(path):
                tar.add(path, arcname=os.path.relpath(path, OUT))
    print(f"wrote {archive}: {len(used)} figures; files with open TODO markers: {len(todo)}")


if __name__ == "__main__":
    main()
