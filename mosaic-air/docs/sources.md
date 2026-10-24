# Data and sources

1. Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5RK5G>. The dataset page documents 420,768 hourly records from 12 Beijing monitoring sites, with pollutant and meteorological variables, March 2013–February 2017, and missing observations. License: CC BY 4.0. <https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality>
2. Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. (2017). Cautionary tales on air-quality improvement in Beijing. *Proceedings of the Royal Society A, 473*(2205), 20170457. <https://doi.org/10.1098/rspa.2017.0457>
3. Tsokov, S., Lazarova, M., & Aleksieva-Petrova, A. (2022). A hybrid spatiotemporal deep model based on CNN and LSTM for air pollution prediction. *Sustainability, 14*(9), 5104. The paper reports longitude/latitude values for the 12 Beijing monitoring sites used to construct `src/stations.json`. <https://doi.org/10.3390/su14095104>
4. Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Software]. <https://open-meteo.com/en/docs/historical-weather-api>. Its hourly historical API provides weather variables including wind speed and direction. It is a candidate source for a future weather-feed extension; it is not used in the benchmark in this repository.

## Coordinate note

The UCI record provides station names but the benchmark CSV does not include latitude and longitude. `src/stations.json` uses approximate published coordinates so the pilot can form spatial edge distances and render the station map. They should be verified against the relevant monitoring authority before operational use.
