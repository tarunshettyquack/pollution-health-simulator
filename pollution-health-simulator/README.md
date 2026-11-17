# Longitudinal Air Pollution & Pharmacological Trade-Off Simulator

An agent-based Monte Carlo simulation modeling how chronic and acute exposure to air pollution ($\text{PM}_{2.5}$, $\text{NO}_2$, $\text{O}_3$, $\text{SO}_2$) degrades lung health over a human lifespan, and how **geographic relocation** and **pharmacological interventions** alter survival curves and life expectancy ($\Delta LE$).

## Key Features
* **Longitudinal Health & Relocation Modeling:** Tracks individual lung health (`0–100`) year-by-year from birth, separating reversible airway inflammation (which recovers upon moving to cleaner air) from permanent structural damage that lowers maximum recoverable lung capacity.
* **Epidemiological Risk Multipliers:** Scales age-specific disease incidence using pollutant relative risks ($RR$) per $+10\ \mu\text{g/m}^3$ above WHO guidelines.
* **Pharmacometric Benefit vs. Toxicity:** Models therapeutic efficacy via saturating $E_{\max}$ dose-response curves while accounting for quadratic dose-dependent side-effect penalties, age sensitivity, and overdose mortality.
* **Automated Data Ingestion (`data_sorter.py`):** Normalizes raw CSV exports from WHO Global Health Observatory life tables, IHME Global Burden of Disease (GBD) datasets, and OpenAQ/CPCB air quality monitors into unified simulation schemas.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the simulation: `python main.py`
3. Results and trajectory plots are saved automatically inside `output/`.

## Current Limitations & Roadmap
* Default parameters in `data/` are illustrative baselines for structural testing, not clinical values. Ongoing work focuses on calibrating permanent lung-ceiling loss against longitudinal cohort studies.