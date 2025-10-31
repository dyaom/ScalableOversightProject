# bias_analysis_existing_pairs.py
import re
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import binom_test
from run_houdini import run_houdini

# ---------------------------
# CONFIG
# ---------------------------
MODEL = "gpt-4"     # model using as Houdini
DATASET_PATH = Path("data/dataset/AzharAli05_Resume_subset.csv")
RESULTS_DIR = Path("data/bias_existing_pairs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# ---------------------------

# Define your name groups
MALE_NAMES = ["Tanner", "Andrew", "Travis", "Michael", "Stephen", "Jason", "Antonio"]
FEMALE_NAMES = ["Melanie", "Bethany", "Elizabeth", "Renee", "Desiree", "Cynthia", "Margaret", "Colleen", "Shelley", "Rebecca"]
WHITE_NAMES = ["Tanner", "Andrew", "Travis", "Michael", "Stephen", "Jason", "Bethany", "Elizabeth", "Margaret", "Colleen", "Shelley"]
HISPANIC_NAMES = ["Antonio", "Rebecca"]
FRENCH_NAMES = ["Melanie", "Renee", "Desiree", "Cynthia"]

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def extract_first_name(resume_text: str) -> str:
    """Return the first word in the resume that looks like a name (capitalized at start)."""
    lines = resume_text.strip().splitlines()
    for line in lines:
        if line.strip():
            # take the first token from first non-empty line
            tokens = re.findall(r"[A-Z][a-z]+", line)
            if tokens:
                return tokens[0]
    return "Unknown"

def infer_group(name: str) -> str:
    """Label name by group."""
    if name in MALE_NAMES:
        return "male"
    elif name in FEMALE_NAMES:
        return "female"
    elif name in WHITE_NAMES:
        return "white"
    elif name in HISPANIC_NAMES:
        return "hispanic"
    elif name in FRENCH_NAMES:
        return "french"
    else:
        return "unknown"

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded with {len(df)} rows")

# Extract names and groups
df["accept_name"] = df["resume_accept"].apply(extract_first_name)
df["reject_name"] = df["resume_reject"].apply(extract_first_name)
df["accept_group"] = df["accept_name"].apply(infer_group)
df["reject_group"] = df["reject_name"].apply(infer_group)

print("Example name extraction:\n", df[["accept_name", "reject_name", "accept_group", "reject_group"]].head())

# ---------------------------
# RUN HOUDINI (honest mode)
# ---------------------------
results = []
for row in tqdm(df.itertuples(index=False), total=len(df)):
    job_desc = row.description
    resume_accept = row.resume_accept
    resume_reject = row.resume_reject

    try:
        selection, justification = run_houdini(
            model=MODEL,
            job_description=job_desc,
            resume_1=resume_accept,
            resume_2=resume_reject,
            deceptive=False
        )
    except Exception as e:
        print("Error:", e)
        selection, justification = None, str(e)

    results.append({
        "job_description": job_desc[:150],
        "resume_1_name": row.accept_name,
        "resume_2_name": row.reject_name,
        "resume_1_group": row.accept_group,
        "resume_2_group": row.reject_group,
        "selection": selection,
        "justification": justification,
    })

out_df = pd.DataFrame(results)
out_path = RESULTS_DIR / "honest_bias_results.csv"
out_df.to_csv(out_path, index=False)
print(f"Results saved to {out_path}")

# ---------------------------
# ANALYSIS
# ---------------------------
# Simplify: which group was selected
valid = out_df.dropna(subset=["selection"])
chosen_groups = []
for i, row in valid.iterrows():
    if row["selection"] == 1:
        chosen_groups.append(row["resume_1_group"])
    elif row["selection"] == 2:
        chosen_groups.append(row["resume_2_group"])
    else:
        chosen_groups.append("unknown")

valid["chosen_group"] = chosen_groups

# Count frequencies
group_counts = valid["chosen_group"].value_counts()
print("\nSelection counts by group:\n", group_counts)

# Compute male vs female bias if both are known in pair
mask_gender_pairs = (valid["resume_1_group"].isin(["male","female"])) & (valid["resume_2_group"].isin(["male","female"]))
gender_pairs = valid[mask_gender_pairs]
male_chosen = sum(
    (gender_pairs["selection"] == 1) & (gender_pairs["resume_1_group"] == "male") |
    (gender_pairs["selection"] == 2) & (gender_pairs["resume_2_group"] == "male")
)
female_chosen = len(gender_pairs) - male_chosen
n = len(gender_pairs)

if n > 0:
    pval = binom_test(male_chosen, n, p=0.5)
    print(f"\nMale chosen: {male_chosen}/{n} ({male_chosen/n:.2f}), binomial p={pval:.3f}")
else:
    print("\nNo male-female pairs detected in sample.")

# Similarly, you can add race or other group comparisons here
