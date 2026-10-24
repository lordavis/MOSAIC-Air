# Presentation guide

## Suggested 8-minute flow

1. **The problem (45 sec):** A city average can hide local pollution peaks. Monitoring teams need short-term forecasts that show where and how uncertain risk may change.
2. **The project (45 sec):** MOSAIC-Air predicts hourly PM₂.₅ at multiple stations. The pilot makes the abstract concrete using the public 12-site Beijing benchmark.
3. **The model (90 sec):** A temporal convolution summarizes each station's recent history. Graph attention lets stations exchange information. Wind direction and speed bias which stations may influence another, with distance also included.
4. **The study design (60 sec):** Use the previous 24 hours, forecast 1/6/24 hours ahead, and split time chronologically. Compare against persistence and HistGradientBoosting.
5. **The result (90 sec):** At 6 hours WAGTF-Net improves MAE to 28.80 µg/m³ from 31.19; at 24 hours it improves modestly to 53.03 from 54.53. At 1 hour, HistGradientBoosting has lower MAE. The model is not best at every horizon.
6. **Uncertainty and missing data (60 sec):** P10–P90 coverage is close to 80% overall, but the 24-hour interval is wide. Masking an additional 20% of recent PM₂.₅ inputs raises one-hour MAE by about 64%, so missing data remain important.
7. **The demo and management use (60 sec):** Show the historical dashboard: choose a replay time, switch horizon, click a station, and inspect predicted concentration, interval, event probability, weather context, and hotspot ranking.
8. **Close (30 sec):** This is a reproducible research prototype. It needs current local data, weather forecasts, held-out-site testing, and field validation before operational use.

## Likely questions

**Why use Beijing for an environmental-management course at Amritapuri?**  
The abstract did not specify a city, and the Beijing benchmark is a documented, multi-site hourly dataset that makes the method reproducible. It is a technical pilot, not evidence about local air quality in Kerala. A local dataset is a key next step.

**What does the 75 µg/m³ threshold mean?**  
It is a project-defined benchmark used to form an episode classification task. It should not be described as a health guideline or local legal threshold.

**Why does the model help at six hours but not one hour?**  
At one hour, pollution is highly persistent, so recent values and a tabular baseline are already strong. At six hours, the temporal and spatial representations may provide more added information in this dataset. This result needs replication and ablation before making a general claim.

**Are the map and alerts live?**  
No. The application replays held-out historical timestamps from 2016–17 to demonstrate the decision-support interface. It does not ingest current observations or issue real alerts.

**How is wind used?**  
The graph attention score is biased toward source-to-destination station bearings aligned with the source station's downwind direction, scaled by wind speed. It is an approximate influence prior, not a physical atmospheric transport simulator.

**What does the uncertainty interval tell us?**  
P10 and P90 form a nominal 80% interval. Aggregate coverage is 80–84% in the standard test, but the 24-hour interval is much wider. Coverage should be checked by site and conditions before anyone uses it operationally.

**What would you do next?**  
Use a local verified dataset, join a forecast weather source, test station holdouts and sensor outages, calibrate intervals per site and horizon, and review alert thresholds with local environmental authorities.
