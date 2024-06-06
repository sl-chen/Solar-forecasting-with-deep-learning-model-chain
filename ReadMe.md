### Improved satellite-based intra-day solar forecasting with a chain of deep learning models

This repository includes the data as well as example scripts for data processing and figure production for the research paper: [https://www.sciencedirect.com/science/article/pii/S0038092X22004236](https://www.sciencedirect.com/science/article/pii/S0196890424005399).

The following figure shows the flowchart for GHI forecasting using spectral satellite images and deep learning model chain, the end-to-end deep learning model, and the hybrid physical deep learning model.

![image](https://github.com/sl-chen/Solar-forecasting-with-deep-learning-model-chain/blob/main/figures/Flowchart.PNG)


#### Data
The satellite data of GOES-16 is downloaded via public available source, e.g., Amazon Web Services, for GOES-16, please refer to: https://docs.opendata.aws/noaa-goes16/cics-readme.html#accessing-goes-data-on-aws.
There are 8 selected spectral bands used: C01, C03, C04, C05, C06, C07, C09, and C11.

|Band|$\lambda$ [&mu m]|Center $\lambda$ [$\mu$m]|Resolution (km)|Type|Valid range|Scale factor|Add offset|
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

#### Strategies for dynamic range, upper and lower bounds determination

| Data type | Description | Time period (year) |
|:-----:|:---------: | :---------: | 
| SURFRAD<sup>a</sup>  | Irradiance measurements   | On-site measurements of solar irradiance  | 2019, 2020| 
| GOES-16<sup>b</sup>  | Satellite measured radiance | Radiance of eight selected spectral bands | 2019, 2020|
| NSRDB<sup>c</sup>  | Derived ground-level irradiance | Satellite-derived irradiance with a physical model | 2020|

* Time window is used to determine the dynamic range, a moving time window means it moves with the time of interest,
so the upper and lower bound will change. While the monthly fixed time window means it is fixed in the month of
interest, so the upper and lower bounds remain constant in the month.

For an example using the four strategies for three bands, please refer to [DRA_ci_calculation_4_strategies.ipynb](https://github.com/sl-chen/GHI-estimation-by-GOES-16/blob/main/DRA_ci_calculation_4_strategies.ipynb)

#### Cloud index (CI) to clear-sky index (CSI) methods
Four methods to convert derived CI to CSI for GHI estimation.
|Method|GHI calculation|
|:-----:|:---------: |
|  1  |  GHI = GHIcs · CSI,<br />CSI = 0.02 + 0.98 · (1 − CI) | 
|  2  |  GHI = CSI · GHIcs · (0.0001 · CSI + 0.9),<br />CSI = 2.36 · CI5 − 6.3 · CI4 + 6.22 · CI3 − 2.63 · CI2 − 0.58 · CI + 1 | 
|  3  |  GHI = GHIcs · CSI<br />CSI = 1.2, CI ≤ −0.2;<br />CSI = 1.0 − CI, −0.2 < CI ≤ 0.8;<br />CSI = 2.0667 − 3.6667 · CI + 1.6667 · CI2, 0.8 < CI ≤ 1.1;<br />CSI = 0.05, 1.1 < CI. | 
|  4  |  GHI = GHIcs · CSICSI = 1.2, CI ≤ −0.2;<br />CSI = 1.0 − CI, −0.2 < CI ≤ 0.8;<br />CSI = 1.1661 − 1.781 · CI + 0.73 · CI2, 0.8 < CI ≤ 1.05;<br />CSI = 0.09, 1.05 < CI.  | 

For the comparison of different CI-to-CSI methods and Strategies for upper and lower bounds, csi calculation, please refer to [DRA_comparison_lb_ub_ci_csi.ipynb](https://github.com/sl-chen/GHI-estimation-by-GOES-16/blob/main/DRA_comparison_lb_ub_ci_csi.ipynb).

The comparison of bands 1, 2, and 3 for GHI estimation is presented in [DRA_c1c2c3_comparison.ipynb](https://github.com/sl-chen/GHI-estimation-by-GOES-16/blob/main/DRA_c1c2c3_comparison.ipynb).

#### Clear-sky models
Four clear-sky models are used in this study, namely, Ineichen-Perez, McClear, REST2, Improved Ineichen-Perez (Ineichen-Perez TL).
|Clear-sky model|Input parameters|Data source|
|:-----:|:---------: | :---------: |
|  Ineichen-Perez  |  $I_0$, $\theta$, $h$, $T_L$ | SoDa database | 
|   McClear  |  $I_0$, $\theta$, $h$, $\rho_g$, $P_a$, $T_a$, $\tau_{550}$, $\alpha$, $u_{O_3}$ , $u_{H_2O}$ | CAMS |
|  REST2  |  $I_0$, $\theta$, $\rho_g$, $P_a$, $\tau_{550}$, $\alpha$, $u_{O_3}$ , $u_{NO_2}$, $u_{H_2O}$ | NSRDB |
|  Ineichen-Perez TL  |  $I_0$, $\theta$, $h$, $P_a$, $T_a$, $\phi$, $V$  | Local measurements| 

Input parameters for the used clear-sky models. The variables are the solar constant $I_0$ [W/m^2], solar zenith angle $\theta$ [degree], altitude $h$ [m], Linke turbidity $T_L$, surface albedo $\rho_g$, local pressure $P_a$ [mb], ambient temperature $T_a$ [K], AOD at 550 nm $\tau_{550}$, Ångström exponent $\alpha$, total ozone amount $u_{O_3}$ [atm-cm], total precipitable water vapor $u_{H_2O}$ [cm], total nitrogen dioxide amount $u_{NO_2}$ [atm-cm], relative humidity $\phi$ [\%], wind speed $V$ [m/s].

Finally, the comparison of different clear-sky models for GHI estimation is shown in [DRA_clear-sky_models_in_GHI_estimation.ipynb](https://github.com/sl-chen/GHI-estimation-by-GOES-16/blob/main/DRA_clear-sky_models_in_GHI_estimation.ipynb).

Note that the examples made here are only for the DRA station. However, the results for other stations follows the same methods and procedure.
