import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_sorter import EpidemiologicalDataSorter

WHO_SAFE_PM25 = 5.0

LOCATIONS = {
    "Clean_Air_City": 8.0,
    "Moderate_Metro": 35.0,
    "Severe_Pollution_Hub": 110.0
}

DRUGS = {
    "none": {"emax_severity": 0.0, "emax_mortality": 0.0, "standard_dose": 1.0, "max_safe_dose": 1.0, "side_effect_prob": 0.0, "side_effect_penalty": 0.0, "overdose_mortality": 0.0},
    "budesonide_inhaled": {"emax_severity": 0.60, "emax_mortality": 0.60, "standard_dose": 400.0, "max_safe_dose": 1600.0, "side_effect_prob": 0.05, "side_effect_penalty": 1.5, "overdose_mortality": 0.01},
    "tiotropium": {"emax_severity": 0.30, "emax_mortality": 0.20, "standard_dose": 18.0, "max_safe_dose": 36.0, "side_effect_prob": 0.08, "side_effect_penalty": 1.5, "overdose_mortality": 0.02},
    "atorvastatin": {"emax_severity": 0.15, "emax_mortality": 0.30, "standard_dose": 40.0, "max_safe_dose": 80.0, "side_effect_prob": 0.03, "side_effect_penalty": 6.0, "overdose_mortality": 0.05}
}

class PersonSimulation:
    def __init__(self, location_schedule, drug_plan, seed=None):
        self.rng = np.random.default_rng(seed)
        self.location_schedule = location_schedule
        self.drug_plan = drug_plan
        self.age = 0
        self.alive = True
        self.lung_health = 100.0
        self.max_lung_ceiling = 100.0
        self.chronic_disease_burden = 0.0
        self.side_effect_damage = 0.0

    def get_current_pm25(self):
        active_loc = "Clean_Air_City"
        for start_age in sorted(self.location_schedule.keys()):
            if self.age >= start_age:
                active_loc = self.location_schedule[start_age]
        return LOCATIONS.get(active_loc, 35.0)

    def get_current_drug(self):
        active_drug, dose = "none", 0.0
        for start_age in sorted(self.drug_plan.keys()):
            if self.age >= start_age:
                active_drug, dose = self.drug_plan[start_age]
        return DRUGS[active_drug], dose

    def step_year(self):
        if not self.alive:
            return

        pm25 = self.get_current_pm25()
        excess_pm25 = max(0.0, pm25 - WHO_SAFE_PM25)
        drug, dose = self.get_current_drug()

        if dose > 0 and drug["standard_dose"] > 0:
            ed50 = drug["standard_dose"] * 0.5
            dose_response = dose / (dose + ed50)
            sev_reduction = drug["emax_severity"] * dose_response
            mort_reduction = drug["emax_mortality"] * dose_response
        else:
            sev_reduction, mort_reduction = 0.0, 0.0

        pollution_rr = 1.012 ** (excess_pm25 / 10.0)
        annual_pollution_wear = (excess_pm25 * 0.015) * (1.0 - sev_reduction)

        if excess_pm25 > 40.0:
            permanent_scarring = (excess_pm25 - 40.0) * 0.004 * (1.0 - sev_reduction)
            self.max_lung_ceiling = max(20.0, self.max_lung_ceiling - permanent_scarring)

        base_incidence = 0.002 + (0.0004 * self.age)
        if self.rng.random() < (base_incidence * pollution_rr):
            self.chronic_disease_burden += 4.0 * (1.0 - sev_reduction)

        recovery = 1.2 * (1.0 + sev_reduction) if (excess_pm25 < 15.0 and self.lung_health < self.max_lung_ceiling) else 0.0

        overdose_hazard = 0.0
        if dose > 0:
            dose_ratio = dose / drug["standard_dose"]
            age_vuln = 1.5 if (self.age < 5 or self.age > 65) else 1.0
            se_prob = min(0.95, drug["side_effect_prob"] * (dose_ratio ** 2) * age_vuln)
            if self.rng.random() < se_prob:
                self.side_effect_damage += drug["side_effect_penalty"] * 0.25
            if dose > drug["max_safe_dose"]:
                overdose_hazard = drug["overdose_mortality"] * (dose / drug["max_safe_dose"])

        age_wear = 0.25 if self.age > 25 else 0.0
        raw_health = self.lung_health - age_wear - annual_pollution_wear - (self.chronic_disease_burden * 0.05) - (self.side_effect_damage * 0.05) + recovery
        self.lung_health = float(np.clip(raw_health, 0.0, self.max_lung_ceiling))

        base_mortality = 0.0005 * np.exp(0.088 * self.age)
        health_deficit_hazard = ((100.0 - self.lung_health) / 100.0) ** 2.2 * 0.12 * (1.0 - mort_reduction)
        total_death_prob = min(1.0, base_mortality + health_deficit_hazard + overdose_hazard)

        if self.rng.random() < total_death_prob or self.lung_health <= 0:
            self.alive = False
        else:
            self.age += 1

def run_cohort(n_people, location_schedule, drug_plan):
    ages_at_death = []
    yearly_health = {age: [] for age in range(100)}
    for i in range(n_people):
        person = PersonSimulation(location_schedule, drug_plan, seed=i)
        while person.alive and person.age < 100:
            yearly_health[person.age].append(person.lung_health)
            person.step_year()
        ages_at_death.append(person.age)
    avg_trajectory = [np.mean(yearly_health[a]) if yearly_health[a] else 0.0 for a in range(85)]
    return round(float(np.mean(ages_at_death)), 2), avg_trajectory

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    if os.path.exists("raw_data"):
        sorter = EpidemiologicalDataSorter(output_dir="data")
        for f in glob.glob("raw_data/*.csv"):
            sorter.ingest_file(f, default_location=os.path.basename(f).split("_")[0].capitalize())

    if os.path.exists("data/locations_pm25.csv"):
        loc_df = pd.read_csv("data/locations_pm25.csv")
        for loc, grp in loc_df.groupby("location"):
            LOCATIONS[loc] = float(grp["pm25"].mean())

    scenarios = {
        "Clean Air Baseline": ({0: "Clean_Air_City"}, {0: ("none", 0)}),
        "Severe Pollution (Lifetime)": ({0: "Severe_Pollution_Hub"}, {0: ("none", 0)}),
        "Relocate to Clean Air at 30": ({0: "Severe_Pollution_Hub", 30: "Clean_Air_City"}, {0: ("none", 0)}),
        "Relocate at 30 + Budesonide": ({0: "Severe_Pollution_Hub", 30: "Clean_Air_City"}, {30: ("budesonide_inhaled", 400)}),
        "Severe Pollution + Overdose": ({0: "Severe_Pollution_Hub"}, {20: ("budesonide_inhaled", 2400)})
    }

    plt.figure(figsize=(10, 6))
    results = []
    for name, (loc_s, drug_s) in scenarios.items():
        le, traj = run_cohort(1500, loc_s, drug_s)
        results.append({"Scenario": name, "Life Expectancy (yrs)": le})
        plt.plot(range(85), traj, label=f"{name} (LE: {le}y)", linewidth=2)

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    df_res.to_csv("output/summary_results.csv", index=False)

    plt.title("Simulated Lung Health Trajectory & Life Expectancy Across Scenarios")
    plt.xlabel("Age (Years)")
    plt.ylabel("Mean Lung Health Score (0 - 100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/lung_health_trajectories.png", dpi=150)
    print("\nSaved chart to output/lung_health_trajectories.png")