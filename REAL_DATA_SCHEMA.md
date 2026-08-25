# Real-data analysis schema

The commercial swine-production observations are proprietary and are not included in this repository. This file documents the 37 variables used after preprocessing so that the public analysis workflow can be inspected without exposing row-level records.

The source file contains 2,592 rows and 43 fields; the complete-case analysis contains 2,556 rows. Thirty-one source fields enter the analysis matrix directly. Four source fields (`PRRSatPlacement`, `Mycoplasma_Status`, `LateralPRRS`, and `Year_Quarter`) are recoded into six indicators (`PRRS_binary`, `MYCO_binary`, `LateralPRRS_binary`, and Q2-Q4). Eight source fields do not enter the 37-variable matrix; their source-file characteristics and dispositions are documented below.

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
| `PWMFyear` | continuous | pigs weaned per mated female per year |
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

## Source-field disposition outside the 37-variable matrix

The following eight authorized source fields do not enter the analysis matrix. These entries record observable source-file characteristics to make preprocessing auditable; they are not a post hoc variable-selection analysis.

| Source field | Analysis disposition | Observed source-file characteristic |
|---|---|---|
| `farrow_sows` | Not included in the 37-variable matrix | Complete numeric field (651 distinct observed values) |
| `PRRSVACCINE` | Not included in the 37-variable matrix | Complete categorical field with three observed levels |
| `WeanPigShot` | Not included in the 37-variable matrix | Categorical field with five observed nonmissing levels; 447/2,592 (17.2%) values missing |
| `SowFarmMed` | Not included in the 37-variable matrix | Categorical field with three observed nonmissing levels; 1,517/2,592 (58.5%) values missing |
| `LateralCoronavirus` | Not included in the 37-variable matrix | Complete categorical field with three observed levels |
| `LateralEnteric` | Not included in the 37-variable matrix | Complete categorical field with four observed levels |
| `OtherHealthIssue` | Not included in the 37-variable matrix | Complete categorical field with two observed levels |
| `PRRS_mshmp3` | Not included separately | Complete two-level field; its epidemic indicator matches `PRRS_binary` for all 2,592 rows |

## Grouping and time-field audit

The source field `Year_Quarter` contains only the values 1-4. It encodes quarter of year and is used to construct Q2-Q4 seasonal indicators; it does not contain calendar year, dates, or a sequential time index. `SowFarmMed` contains medication categories (CTC, Linco, and Mixed), not a sow-farm identifier. Other name-matched fields are production measurements or rates, not dates or grouping identifiers. The authorized analysis file therefore contains no farm, site, batch, or other defensible operational grouping field. The workflow does not impose temporal constraints or report a blocked/cluster bootstrap. Its effective-sample-size analysis is labeled as an illustrative sensitivity, not cluster-corrected inference.

## Mixed-variable diagnostic

In the pinned 185-edge NOTEARS candidate, all six binary indicator nodes had in-degree zero. Thus every edge considered by the reported local-BIC refinements had a continuous child, so Gaussian local BIC was not used as a binary-response likelihood on those paths. This fact does not validate the mixed-variable NOTEARS initialization; it remains a working model, and type-specific decomposable scores are the natural extension when binary children occur.
