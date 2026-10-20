# SandMuon paper and evidence

The preliminary paper is [main.pdf](main.pdf), with LaTeX source in [main.tex](main.tex). Author: Hamza Kharbouch. Code and compact reproduction artifacts: https://github.com/pxike/SandMuon. The standalone package is also at `/home/hamzakharbouch/SandMuon`. No submission, venue, or DOI is claimed.

The paper reuses the archived S8/Muon/PE reference records, documents the completed 103M SAND run, reports the separate FP32 geometry screen, and includes two application demonstrations. The main run completed 15,106 updates under a 16,000-update cap and the PE training-time budget. The cross-host complete-recipe comparison does not isolate geometry.

## Build

From this directory, run `tectonic -X compile main.tex`. With a conventional TeX installation, run `pdflatex main`, `bibtex main`, then `pdflatex main` twice.

Rebuild both figures using `python tools/build_evidence.py --plots-only` with matplotlib installed. That mode reads only bundled data. The default builder performs archive extraction and therefore requires the original ODL checkout; it is not needed to build the released paper.

[The evidence audit](evidence_audit.md) lists protocol differences, source integrity, and application limitations. `data/summary.json` and `data/main_results.csv` provide machine-readable results. Full local records also retain update and learning-rate traces; the public bundle omits those larger redundant traces and includes the validation records needed for plotting. No datasets or model checkpoints are distributed.
