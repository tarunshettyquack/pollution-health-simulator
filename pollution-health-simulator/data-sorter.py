import os
import re
import pandas as pd
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

class EpidemiologicalDataSorter:
    def __init__(self, output_dir=DATA_DIR):
        self.output_dir = output_dir

    def ingest_file(self, filepath, default_location="Unknown_City"):
        df = pd.read_csv(filepath)
        cols_lower = {c.lower().strip(): c for c in df.columns}
        print(f"\n[INGEST] Inspecting '{filepath}' ({len(df)} rows)...")

        if any(k in cols_lower for k in ["pm25", "pm2.5", "parameter.name", "pollutant"]):
            return self._process_air_quality(df, cols_lower, default_location)
        elif any(k in cols_lower for k in ["cause_name", "measure_name", "val"]):
            return self._process_ihme_gbd(df, cols_lower)
        elif any(k in cols_lower for k in ["nmx", "nqx", "indicatorcode", "numericvalue", "spatialdim"]):
            return self._process_who_lifetable(df, cols_lower)
        else:
            raise ValueError(f"Unrecognized dataset format in {filepath}.")

    def _process_air_quality(self, df, cols_lower, default_location):
        date_col = next((cols_lower[k] for k in cols_lower if "date" in k or "time" in k or "year" in k), None)
        if date_col:
            df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
            df["year"] = df["year"].fillna(pd.to_numeric(df[date_col], errors="coerce"))
        else:
            df["year"] = 2024

        loc_col = next((cols_lower[k] for k in cols_lower if k in ["city", "location", "station", "country"]), None)
        df["location"] = df[loc_col].fillna(default_location) if loc_col else default_location

        param_col = next((cols_lower[k] for k in cols_lower if "parameter" in k or "pollutant" in k), None)
        val_col = next((cols_lower[k] for k in cols_lower if k in ["value", "val", "concentration"]), None)

        if param_col and val_col:
            df[param_col] = df[param_col].str.lower().str.replace(".", "", regex=False)
            df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
            df = df[df[val_col] >= 0]
            pivoted = df.pivot_table(index=["location", "year"], columns=param_col, values=val_col, aggfunc="mean").reset_index()
        else:
            rename_map = {c_orig: c_low.replace(".", "").replace("_", "") for c_low, c_orig in cols_lower.items() if c_low.replace(".", "").replace("_", "") in ["pm25", "no2", "o3", "so2"]}
            pivoted = df.rename(columns=rename_map)
            for p in ["pm25", "no2", "o3", "so2"]:
                if p not in pivoted.columns:
                    pivoted[p] = np.nan
                pivoted[p] = pd.to_numeric(pivoted[p], errors="coerce")
                pivoted.loc[pivoted[p] < 0, p] = np.nan
            pivoted = pivoted.groupby(["location", "year"], as_index=False)[["pm25", "no2", "o3", "so2"]].mean()

        for p in ["pm25", "no2", "o3", "so2"]:
            if p not in pivoted.columns:
                pivoted[p] = 0.0
            pivoted[p] = pivoted[p].fillna(0.0).round(2)

        pivoted = pivoted.dropna(subset=["year"])
        pivoted["year"] = pivoted["year"].astype(int)
        out_df = pivoted[["location", "year", "pm25", "no2", "o3", "so2"]]
        self._merge_and_save(out_df, os.path.join(self.output_dir, "locations_pm25.csv"), ["location", "year"])
        return out_df

    def _process_ihme_gbd(self, df, cols_lower):
        if "measure_name" in cols_lower:
            df = df[df[cols_lower["measure_name"]].str.contains("Incidence", case=False, na=False)]
        if "metric_name" in cols_lower:
            df = df[df[cols_lower["metric_name"]].str.contains("Rate", case=False, na=False)]

        cause_col = cols_lower.get("cause_name", list(df.columns)[0])
        age_col = cols_lower.get("age_name", None)
        val_col = cols_lower.get("val", None)

        df["name"] = df[cause_col].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
        df["annual_prob"] = pd.to_numeric(df[val_col], errors="coerce") / 100000.0

        def map_age_bucket(age_str):
            nums = [int(n) for n in re.findall(r"\d+", str(age_str))]
            if not nums: return "15-44"
            if nums[0] < 5: return "0-4"
            elif nums[0] < 15: return "5-14"
            elif nums[0] < 45: return "15-44"
            elif nums[0] < 65: return "45-64"
            else: return "65+"

        df["age_bucket"] = df[age_col].apply(map_age_bucket) if age_col else "15-44"
        grouped = df.groupby(["name", "age_bucket"], as_index=False)["annual_prob"].mean()
        pivoted = grouped.pivot(index="name", columns="age_bucket", values="annual_prob").reset_index()

        for bucket in ["0-4", "5-14", "15-44", "45-64", "65+"]:
            pivoted[f"inc_{bucket}"] = pivoted[bucket].fillna(0.0).round(6) if bucket in pivoted.columns else 0.0

        defaults = {"chronic": 1, "severity": 15.0, "case_fatality": 0.02, "smoking_rr": 1.5, "rr_pm25": 1.10, "rr_no2": 1.03, "rr_o3": 1.02, "rr_so2": 1.02}
        for col, default_val in defaults.items():
            if col not in pivoted.columns:
                pivoted[col] = default_val

        final_cols = ["name", "chronic", "severity", "case_fatality", "smoking_rr", "rr_pm25", "rr_no2", "rr_o3", "rr_so2", "inc_0-4", "inc_5-14", "inc_15-44", "inc_45-64", "inc_65+"]
        out_df = pivoted[final_cols]
        self._merge_and_save(out_df, os.path.join(self.output_dir, "diseases.csv"), ["name"])
        return out_df

    def _process_who_lifetable(self, df, cols_lower):
        val_col = next((cols_lower[k] for k in cols_lower if k in ["numericvalue", "nmx", "nqx", "value"]), None)
        age_col = next((cols_lower[k] for k in cols_lower if k in ["dim2", "agegroup", "age"]), None)
        df["mortality"] = pd.to_numeric(df[val_col], errors="coerce")

        def parse_who_age(age_code):
            nums = [int(n) for n in re.findall(r"\d+", str(age_code))]
            if not nums or nums[0] < 5: return ("0-4", 0, 4, 1.4, 1.5)
            elif nums[0] < 15: return ("5-14", 5, 14, 0.8, 1.2)
            elif nums[0] < 45: return ("15-44", 15, 44, 1.0, 1.0)
            elif nums[0] < 65: return ("45-64", 45, 64, 1.3, 1.1)
            else: return ("65+", 65, 120, 1.8, 1.5)

        parsed = df[age_col].apply(parse_who_age) if age_col else [("15-44", 15, 44, 1.0, 1.0)] * len(df)
        df[["age_group", "min_age", "max_age", "frailty", "drug_sensitivity"]] = pd.DataFrame(parsed.tolist(), index=df.index)

        out_df = df.groupby(["age_group", "min_age", "max_age", "frailty", "drug_sensitivity"], as_index=False)["mortality"].mean()
        out_df = out_df.rename(columns={"mortality": "base_mortality"}).sort_values("min_age")
        out_df["base_mortality"] = out_df["base_mortality"].round(6)
        self._merge_and_save(out_df, os.path.join(self.output_dir, "age_groups.csv"), ["age_group"])
        return out_df

    def _merge_and_save(self, new_df, out_path, key_cols):
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset=key_cols, keep="last")
        else:
            combined = new_df
        combined.to_csv(out_path, index=False)
        print(f" -> Saved {len(combined)} clean records to '{out_path}'.")