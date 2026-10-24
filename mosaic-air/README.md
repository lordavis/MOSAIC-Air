# MOSAIC-Air

**Wind-Aware Spatiotemporal Intelligence for Urban Air-Pollution Forecasting**

This project turns the supplied abstract into a reproducible Beijing pilot: an hourly PM₂.₅ forecaster for 12 monitoring stations, a wind-conditioned graph/temporal model, event-probability and uncertainty outputs, two baselines, a REST API, and a small browser dashboard.

The pilot uses the UCI Beijing Multi-Site Air Quality dataset (March 2013–February 2017). Its results are a historical research backtest. The included dashboard replays the held-out 2016–17 period; it is not a live pollution service or a health advisory.

## Run it

Use Python 3.10 or later. Install dependencies, prepare the data, and run the experiment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data
python -m src.pipeline
```

The UCI archive is already included in `data/raw/`. If it is removed, `src.data` downloads it from the official UCI page. The pipeline writes the trained model to `models/wagtf_net.pt` and metrics/figures to `results/`.

Start the historical replay application after training:

```bash
uvicorn src.api:app --reload
```

Open <http://127.0.0.1:8000>. The API exposes `/api/health`, `/api/metadata`, and `/api/forecast?timestamp=2016-12-01T06:00`.

To regenerate the editable report and slides, install `requirements-build.txt` and run `python -m src.build_deliverables`. PDF export uses LibreOffice.

## Project contents

- `src/data.py` downloads, parses, imputes, scales, and saves the station series.
- `src/model.py` defines WAGTF-Net: temporal convolutions plus multi-head graph attention biased by downwind alignment and station distance.
- `src/pipeline.py` trains/evaluates the model, compares persistence and HistGradientBoosting, selects event cutoffs on validation data, and runs a 20% missing-input stress experiment.
- `src/api.py` serves model forecasts and the dashboard.
- `static/index.html` is the responsive historical replay interface.
- `docs/project_report.md` is the full project report source; `docs/project_report.docx` and `.pdf` are presentation-ready versions.
- `MOSAIC-Air_Presentation.pptx` and `.pdf` are the slide deck.
- `docs/presentation_guide.md` contains a short talk track and likely questions.
- `docs/sources.md` records data, coordinate, and future weather-feed sources.

## Evaluation summary

The test period is July 2016–February 2017. With a 24-hour input window and 6-hour-spaced forecast issue times, WAGTF-Net reduced PM₂.₅ MAE against HistGradientBoosting from 31.19 to 28.80 µg/m³ at 6 hours and from 54.53 to 53.03 µg/m³ at 24 hours. At one hour, the gradient-boosted baseline had lower MAE (10.07 versus 10.93 µg/m³), while WAGTF-Net had lower RMSE. Results, thresholds, and uncertainty coverage are in `results/metrics.csv`.

High-pollution episodes are defined for this project as PM₂.₅ ≥ 75 µg/m³. This is an operational experiment threshold, not a statement about health guidance or regulatory compliance. Event F1 thresholds were tuned on the validation period and then frozen for the test period.

## Reproducibility and limitations

The splits are chronological: training through December 2015, validation January–June 2016, and test July 2016–February 2017. Feature scaling uses training data only. Missing input features are forward-filled for at most three hours, then filled with training medians; missingness flags are retained. Missing targets are excluded from loss and metrics.

Site coordinates are approximate values transcribed from published station tables. The benchmark tests temporal generalization across all 12 known stations; it does not test transfer to an unseen city or unseen station. The weather observations in the benchmark are historical, so live deployment would need a forecast weather feed, renewed air-quality observations, monitoring/alert validation, and local environmental review.

## License and attribution

Dataset: Song Chen, *Beijing Multi-Site Air Quality*, UCI Machine Learning Repository, DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0. See `docs/sources.md` for the full citation and station-coordinate references.
