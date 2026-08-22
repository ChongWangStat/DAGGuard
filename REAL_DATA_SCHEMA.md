# Real-data analysis schema

The commercial swine-production observations are proprietary and are not included in this repository. This file documents the 37 variables used after preprocessing so that the public analysis workflow can be inspected without exposing row-level records.

The complete-case analysis in the manuscript contained 2,556 rows. The six indicator variables are constructed by `additional_validation_and_realdata.py`; all remaining variables are numeric production measures retained from the authorized source file.

| Variable | Analysis type | Description / construction |
|---|---|---|
| `PRRS_binary` | binary | 1 when PRRS status at placement is epidemic; 0 otherwise |
| `MYCO_binary` | binary | 1 when *Mycoplasma hyopneumoniae* status is endemic; 0 otherwise |
| `LateralPRRS_binary` | binary | 1 for a lateral PRRSV transmission event; 0 otherwise |
| `Q2` | binary | placement in April-June; Q1 is the seasonal baseline |
| `Q3` | binary | placement in July-September; Q1 is the seasonal baseline |
| `Q4` | binary | placement in October-December; Q1 is the seasonal baseline |
| `Avg_parity_farrow` | continuous/numeric | derived four-level average-parity-at-farrow covariate in the source data |
| `Litters_female_year` | continuous | litters per female per year |
| `mated_inventory_20wks` | continuous | mated-female inventory measure over the source-data 20-week window |
| `PWMFyear` | continuous | pre-weaning mortality factor |
| `nonproductive_days` | continuous | nonproductive days |
| `number_services` | continuous | number of breeding services |
| `wean_to_service` | continuous | weaning-to-service interval |
| `abortions_rate` | continuous | abortion rate |
| `Total_born_avg` | continuous | mean total piglets born per litter |
| `Stillborn_avg` | continuous | mean stillborn piglets per litter |
| `Mummies_avg` | continuous | mean mummified fetuses per litter |
| `prenatal_losses_avg` | continuous | prenatal losses per litter |
| `Born_alive_avg` | continuous | mean piglets born alive per litter |
| `Gestation_days` | continuous | gestation duration |
| `Interval_farrows` | continuous | interval between successive farrowings |
| `Pre_weaning_mortality` | continuous | pre-weaning mortality rate |
| `PWSow` | continuous | pigs weaned per sow |
| `productive_days_rate` | continuous | productive-days rate |
| `services_per_inventory_N_rate` | continuous | breeding services per sow inventory |
| `repeats__rate` | continuous | repeated-service rate |
| `gilts_bred_rate` | continuous | replacement-gilt breeding rate |
| `Last_week_wean_bred_rate` | continuous | wean-to-bred rate in the week preceding placement |
| `pregnant_105days_rate` | continuous | pregnancies confirmed at 105 days |
| `Cull_rate_annual` | continuous | annual sow culling rate |
| `Sow_Death_rate` | continuous | annual sow mortality rate |
| `avg_parity_at_farrow` | continuous | record-level average parity of sows at farrowing |
| `Lactation_days` | continuous | lactation duration |
| `final_inventory` | continuous | final sow/female inventory in the system |
| `Farrowing__rate` | continuous | farrowing rate |
| `HeadIn` | continuous | pigs placed into the nursery |
| `mortality_60days` | continuous | nursery mortality during the first 60 days post-placement |

## Mixed-variable diagnostic

In the 190-edge initial NOTEARS graph reported in the manuscript, all six binary indicator nodes had in-degree zero. Thus every edge considered by NOTEARS-BP had a continuous child, so the Gaussian local BIC was not used as a binary-response likelihood on the reported pruning path. This fact does not validate the mixed-variable NOTEARS initialization; the manuscript therefore also reports a continuous-variable-only sensitivity analysis.
