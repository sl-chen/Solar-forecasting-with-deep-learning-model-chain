### Estimation of high-resolution solar irradiance data using optimized semi-empirical satellite method and GOES-16 imagery

This repository includes the data as well as example scripts for data processing and figure production for the research paper: https://www.sciencedirect.com/science/article/pii/S0038092X22004236.

The following figure shows the flowchart for GHI estimation using semi-emipirical satellite method with several clear-sky models.

![image](https://user-images.githubusercontent.com/54800388/187680737-70c8ddb3-ab49-487a-acc1-d84fc631fba0.png)


#### Data
The satellite data of GOES-16 is downloaded via public available source, e.g., Amazon Web Services, for GOES-16, please refer to: https://docs.opendata.aws/noaa-goes16/cics-readme.html#accessing-goes-data-on-aws.

The used ground data is from SURFRAD stations (see the following table) with quality control.
The snow-free period is obtained using the data from National Aeronautics and Space Administration (NASA) National Snow and Ice Data Center (NSIDC): https://nsidc.org/data/nsidc-0719/versions/1.  
|Station|Latitude (°)|Longitude (°)|Altitude (m)|Timezone|Snow-free period|
|:-----:|:---------: | :---------: | :--------: |:------:| :------------: |
|  BON  |  40.05     | -88.37      |  230       |  UTC-6 | 2019-04-01 - 2019-10-31 |
|  DRA  |  36.62     | -116.02     |  1007      |  UTC-8 | 2019-01-01 - 2019-12-31 |
|  FPK  |  48.31     | -105.10     |  634       |  UTC-7 | 2019-05-03 - 2019-09-30 |
|  GWN  |  34.25     | -89.87      |  98        |  UTC-6 | 2019-01-01 - 2019-12-31 |
|  PSU  |  40.72     | -77.93      |  376       |  UTC-5 | 2019-04-01 - 2019-10-31 |
|  SXF  |  43.73     | -96.92      |  473       |  UTC-6 | 2019-05-01 - 2019-09-30 |
|  TBL  |  40.12     | -105.24     |  1689      |  UTC-7 | 2019-05-02 - 2019-09-30 |

#### Strategies for dynamic range, upper and lower bounds determination

|Strategy|Time window*|Upper bound|Lower bound |
|:-----:|:---------: | :---------: | :--------:|
|  1  |  90 days (moving) | mean of 20 highest values|mean of 40 lowest values| 
|  2  |  60 days (moving) | mean of 20 highest values|mean of 40 lowest values|
|  3  |  30 days (moving) | mean of 10 highest values|mean of second to fifth lowest values|
|  4  |  1 month (fixed)  | mean of 10 highest values|mean of second to fifth lowest values| 

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
