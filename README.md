# AstroSense: Machine Learning for Autonomous Spacecraft Diagnostics

Desktop app (Python + Tkinter) that learns what *healthy* spacecraft telemetry looks like and flags anything unusual.
Mini Project Phase-I, B.Tech Artificial Intelligence, Usha Mittal Institute of Technology, SNDT Women's University, Mumbai (2026-2027).

**Team:** Anushka Waghaye (65), Ananya Shetiya (57), Swara Lele (26)  **Guide:** Prof. Supriya Ingle

## What it does
1. Cleans broken rows (missing or non-numeric values).
2. Min-Max scales every sensor to 0-1.
3. Trains an **Isolation Forest** on healthy telemetry (unsupervised).
4. Scores an uploaded telemetry log and shows a **System Health Score (0-100%)**.
5. Produces a **warning checklist** naming the sensors that deviate most, e.g. "Check Engine - Overheating".

Also: manual single-reading check (sliders), health-over-time curve, real-data evaluation on OPSSAT-AD.

## Quick start
```bash
git clone <your-repo-url> && cd AstroSense
pip install -r requirements.txt
python main.py
```
Tkinter ships with Python on Windows/macOS. On Linux: `sudo apt install python3-tk`.

Demo: *Data* tab > **Upload log CSV...** > `data/samples/sample_telemetry_log.csv`.
Real data: **File > Load OPS-SAT-AD dataset.csv** (see `data/README.md`).

## Repository layout
```
astrosense/backend.py     sensor table, physics-based simulator, Isolation Forest detector (no GUI)
astrosense/app.py         Tkinter interface
main.py                   entry point
scripts/generate_samples.py   regenerates data/samples and docs/SOURCES.md
data/                     sample CSVs + instructions for real datasets
docs/                     synopsis, Phase-I report, SOURCES.md (ranges, equations, references)
```

## Method notes and limitations (please read)
- **Health score:** piecewise-linear map of the Isolation Forest score: 100 up to the 95th percentile of healthy scores, 80 at the flag threshold, 0 at score 1. System score = mean record health x share of unflagged records. This mapping is our design choice, not from a paper.
- **Single-sensor guard:** Isolation Forest is weak when only one of many sensors goes bad, so a record is also flagged if any sensor is 7 or more standard deviations from the healthy mean.
- **Sensor ranges** come from NASA / ESA / vendor documents listed in `docs/SOURCES.md`. Engine temperature, avionics temperature and gyro drift are **unsourced assumptions**: replace them from your hardware datasheets.
- **Simulated data is not real telemetry.** The built-in baseline and sample CSVs come from simplified physics equations.
- A fault that stays inside the healthy range (for example a slow propellant leak that only moves pressure along its normal blow-down curve) cannot be caught by this model.
- Rules of thumb for hydrazine, cabin and radiation figures are for a prototype, not flight-certified limits.

## Key references
- F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation forest," ICDM 2008.
- F. Pedregosa et al., "Scikit-learn," JMLR 12, 2011.
- B. Ruszczak, K. Kotowski, D. Evans, J. Nalepa, "The OPS-SAT benchmark for detecting anomalies in satellite telemetry," Scientific Data, 2025.
- K. Kotowski et al., "European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry," arXiv:2406.17826, 2024.
- NASA-STD-3001 Vol. 2 technical briefs (OCHMO); NASA/TP-20250000142; NASA/TM-2003-212304; ArianeGroup PM25 datasheet.
Full list with the value each source supports: `docs/SOURCES.md`; literature survey: `docs/Phase1_Report.docx`.

## License
MIT (see `LICENSE`).
