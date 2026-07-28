# SENTINEL — AI-Powered Behavioral Anomaly Detection for Cybersecurity

## Setup

```bash
pip install numpy pandas faker scikit-learn pyarrow torch lightgbm shap streamlit plotly joblib
```

## Run, in order

```bash
python data_generator.py --entities 120 --days 60 --attack-rate 0.035   # -> data/events.parquet, data/events.csv
python profiler.py                                                       # -> data/features.parquet
python detector.py                                                       # -> data/scored_events.parquet, data/ae_model.pt
python classifier.py                                                     # -> data/gbm_classifier.joblib
python explain.py                                                        # -> data/sample_alerts.json (smoke test)
python evaluate.py                                                       # -> data/report_metrics_block.md
streamlit run dashboard.py                                               # live SOC console
```

Each stage reads the previous stage's parquet output from `data/`, so they
must be run in this order the first time. After that, `dashboard.py` and
`evaluate.py` can be re-run independently as long as `data/scored_events.parquet`
and `data/gbm_classifier.joblib` exist.

## Files

| File | Purpose |
|---|---|
| `data_generator.py` | Synthetic behavioral telemetry + 7 attack-type injection, batch + streaming modes |
| `profiler.py` | Per-entity/peer-group rolling behavioral profiles, cold-start fallback, engineered deviation features |
| `detector.py` | GRU-Autoencoder (sequence-aware) + Isolation Forest (baseline) unsupervised detectors |
| `classifier.py` | LightGBM attack-type classifier, trained only on flagged events |
| `explain.py` | SHAP-based structured, human-readable alert generation |
| `dashboard.py` | Streamlit SOC analyst console (dark mode, live-feel streaming, drill-down) |
| `evaluate.py` | PR-AUC / precision-recall-at-budget / ablation / per-class accuracy |
| `REPORT.md` | Full write-up: assumptions, architecture, metrics, limitations, future work |
| `SLIDE_DECK_OUTLINE.md` | 12-slide presentation outline |
| `data/events.csv` | Sample generated dataset (labeled) for reference |
| `data/sample_alerts.json` | Sample structured, explained alerts |

See `REPORT.md` for full metrics, an honest ablation (Isolation Forest
currently edges out the GRU-Autoencoder at this data scale, with an
explanation of why), and known limitations.
