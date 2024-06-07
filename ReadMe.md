### Improved satellite-based intra-day solar forecasting with a chain of deep learning models

This repository includes the data as well as example scripts for data processing and figure production for the research paper: [https://www.sciencedirect.com/science/article/pii/S0038092X22004236](https://www.sciencedirect.com/science/article/pii/S0196890424005399).

The following figure shows the flowchart for GHI forecasting using spectral satellite images and deep learning model chain, the end-to-end deep learning model, and the hybrid physical deep learning model.

![image](https://github.com/sl-chen/Solar-forecasting-with-deep-learning-model-chain/blob/main/figures/Flowchart.PNG)


#### Data
The satellite data of GOES-16 is downloaded via public available source, e.g., Amazon Web Services, for GOES-16, please refer to: https://docs.opendata.aws/noaa-goes16/cics-readme.html#accessing-goes-data-on-aws.
There are 8 selected spectral bands used: C01, C03, C04, C05, C06, C07, C09, and C11.

|Band|$\lambda$ [μm]|Center $\lambda$ [μm]|Resolution (km)|Type|Valid range|Scale factor|Add offset|
|:-----:|:---------: | :---------: | :--------: |:------:| :------------: | :------------: | :------------: |
|  1  |  0.45-0.49   | 0.47  | 1 | Near-Infrared | 0-2046  | 0.0707 | -4.5224  |
|  3  |  0.846-0.885 | 0.865 | 1 | Near-Infrared | 0-1022  | 0.3769 | -20.2899 |
|  4  |  1.371-1.386 | 1.378 | 2 | Near-Infrared | 0-2046  | 0.0707 | -4.5224  | 
|  5  |  1.58-1.64   | 1.61  | 1 | Near-Infrared | 0-1022  | 0.0958 | -3.0596  |
|  6  |  2.225-2.275 | 2.25  | 2 | Near-Infrared | 0-1022  | 0.0301 | -0.9610  |
|  7  |  3.80-4.00   | 3.90  | 2 | Infrared      | 0-16382 | 0.0016 | -0.0376  |
|  9  |  6.75-7.15   | 6.95  | 2 | Infrared      | 0-2046  | 0.0225 | -0.8236  |
|  11 |  8.30-8.70   | 8.50  | 2 | Infrared      | 0-4094  | 0.0334 | -1.3022  |


The used ground data is from SURFRAD stations (see the following table) with quality control.

|Station|Latitude (°)|Longitude (°)|Altitude (m)|Timezone|
|:-----:|:---------: | :---------: | :--------: |:------:|
|  BON  |  40.05     | -88.37      |  230       |  UTC-6 |
|  DRA  |  36.62     | -116.02     |  1007      |  UTC-8 |
|  FPK  |  48.31     | -105.10     |  634       |  UTC-7 |
|  GWN  |  34.25     | -89.87      |  98        |  UTC-6 |
|  PSU  |  40.72     | -77.93      |  376       |  UTC-5 |
|  SXF  |  43.73     | -96.92      |  473       |  UTC-6 |
|  TBL  |  40.12     | -105.24     |  1689      |  UTC-7 |

#### A summary of the publicly available data

| Data type | Description | Time period (year) |
|:-----:|:---------: | :---------: | 
| SURFRAD<sup>a</sup>  | Irradiance measurements   | On-site measurements of solar irradiance  | 2019, 2020| 
| GOES-16<sup>b</sup>  | Satellite measured radiance | Radiance of eight selected spectral bands | 2019, 2020|
| NSRDB<sup>c</sup>  | Derived ground-level irradiance | Satellite-derived irradiance with a physical model | 2020|

<sup>a</sup> Available at (https://gml.noaa.gov/grad/surfrad/), can be downloaded by SolarData [1].

<sup>b</sup> Available at (https://registry.opendata.aws/noaa-goes/), can be download by [GOES-2-go](https://github.com/blaylockbk/goes2go).

<sup>c</sup> Available at (https://nsrdb.nrel.gov/), can be downloaded by SolarData.

#### Satellite-derived spatial GHI with deep learning

The following figure shows an illustration of regional solar irradiance estimations for TBL station using spectral satellite images. (a) The target station and $11\times11$ pixel grid of satellite images for single-station solar irradiance estimation. (b) The target station with 121 surrounding locations and the domain of used spectral satellite images for regional solar irradiance estimation.

![image](https://github.com/sl-chen/Solar-forecasting-with-deep-learning-model-chain/blob/main/figures/Region.PNG)

#### Forecasting methods

The following figure shows the strucuture of deep learning models for GHI forecasting with different inputs.

![image](https://github.com/sl-chen/Solar-forecasting-with-deep-learning-model-chain/blob/main/figures/Method.PNG)

The end-to-end deep learning model $\mathbb{F}$ can produce multiple CSI forecasts (multiple-output model) with forecast horizons ($\Delta t$) up to 180 minutes (i.e., 15, 30, 45, 60, 90, 120, 150, and 180-minute), which can be formulated as,
```math
\hat{I}_{t_0+15}, \hat{I}_{t_0+30}, ..., \hat{I}_{t_0+180} = \mathbb{F}({\boldsymbol{x}_{t_0}, \boldsymbol{x}_{t_0-5}, ..., \boldsymbol{x}_{t_0-60}})
```
where $\hat{I}$ denotes the CSI forecast, $\boldsymbol{x}$ represents the spatio-temporal spectral satellite measurements.

The hybrid physical-deep learning model is:
```math
\hat{I}_{t_0+15}, \hat{I}_{t_0+30}, ..., \hat{I}_{t_0+180} = \mathbb{F}(\mathbb{P}({\boldsymbol{x}_{t_0}, \boldsymbol{x}_{t_0-5}, ..., \boldsymbol{x}_{t_0-60}})),
```
where $\mathbb{P}$ is the physical model used to convert spectral satellite measurements $\boldsymbol{x}$ to spatial GHI estimations of NSRDB.

The deep learning model chain is then formulated as:
```math
\hat{I}_{t_0+15}, \hat{I}_{t_0+30}, ..., \hat{I}_{t_0+180} = \mathbb{F}(\mathbb{E}({\boldsymbol{x}_{t_0}, \boldsymbol{x}_{t_0-5}, ..., \boldsymbol{x}_{t_0-60}})),
```
where $\mathbb{E}$ is the deep learning model to derive spatial GHI estimates of SAT-DL from spectral satellite measurements $\boldsymbol{x}$.

#### Example data and notebooks

The spectral satellite data of GOES-16 for BON is available [here](https://drive.google.com/drive/folders/1oUjJ_2rKpEEG6TIbKOHX7C1zAueWnucN?usp=sharing).

The NSRDB data for BON is available [here](https://drive.google.com/drive/folders/12n7YmZbkDdZkt_WcykwvgnRvsx6Eo-16?usp=sharing), satellite-derived spatial GHI using deep learning can be found at [here](https://drive.google.com/drive/folders/1to2rdRhWoN1jdBqllqGgQE7dd6_m8zbW?usp=sharing).

The notebook of the end-to-end deep learning model for satellite-based solar forecasting up to 3 hours is available [here](https://github.com/sl-chen/Solar-forecasting-with-deep-learning-model-chain/blob/main/ghi_forecasting_bon_sat_3h.ipynb).

The notebook of the hybrid physical deep learning model is available [here](https://github.com/sl-chen/Solar-forecasting-with-deep-learning-model-chain/blob/main/ghi_forecasting_bon_nsrdb-3h.ipynb).

The deep learning model using satellite-derived spatial GHI estimates (with a pre-trained deep learning model) thus forms a deep learning model chain, the notebook is available [here](https://github.com/sl-chen/Solar-forecasting-with-deep-learning-model-chain/blob/main/ghi_forecasting_bon_sat-dl-3h.ipynb).


#### Results

The forecast skills of SDL (the deep learning model chain), NS (the hybrid physical deep learning model), and SAT (the end-to-end deep learning model) are shown in the following figure for all the SURFRAD stations.

![image](https://github.com/sl-chen/Solar-forecasting-with-deep-learning-model-chain/blob/main/figures/Skill.PNG)



Note that the examples made here are only for the BON station. However, the results for other stations follows the same methods and procedure.

#### References
[1] Yang, D. (2018). SolarData: An R package for easy access of publicly available solar datasets. Solar Energy, 171, A3-A12.
