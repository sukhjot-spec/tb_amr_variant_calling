import sys
import time
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm

RUN_FILE = "data/african_accessions.txt"
CSV_FILE = "data/sra_master_metadata.csv"
OUTPUT_CSV = "filtered_runs_with_country.csv"
BATCH_SIZE = 500                     # POST can handle larger batches
SLEEP_TIME = 0.5                     # between batches
MAX_RETRIES = 3

AFRICAN_COUNTRIES = {
    "south africa", "kenya", "uganda", "nigeria", "ethiopia",
    "tanzania", "mozambique", "zambia", "zimbabwe", "malawi",
    "ghana", "rwanda", "cameroon", "congo", "somalia",
    "eswatini", "swaziland", "lesotho", "botswana", "namibia",
    "senegal", "mali", "burkina faso", "benin", "togo",
    "ivory coast", "côte d'ivoire", "liberia", "sierra leone",
    "guinea", "gambia", "mauritania", "niger", "chad", "sudan",
    "south sudan", "eritrea", "djibouti", "somaliland",
    "comoros", "madagascar", "mauritius", "seychelles",
    "angola", "zambia", "zimbabwe", "botswana", "namibia", "eswatini"
}

# Expanded attribute names for country
COUNTRY_ATTR_NAMES = [
    "geo_loc_name",
    "country",
    "isolation_source_country",
    "geographic location",
    "sample_location",
    "host_geo_loc_name",
    "location"
]

def is_african(country_str):
    if not isinstance(country_str, str):
        return False
    lower = country_str.lower()
    return any(africa in lower for africa in AFRICAN_COUNTRIES)

def fetch_biosample_metadata(biosample_ids):
    """
    Fetch metadata for a list of BioSample IDs using POST to avoid URI length limits.
    """
    if not biosample_ids:
        return {}
    
    ids = ",".join(biosample_ids)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    data = {
        "db": "biosample",
        "id": ids,
        "retmode": "xml"
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, data=data, timeout=60)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"Request failed (attempt {attempt+1}): {e}. Retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    else:
        # All retries exhausted
        return {bid: {"country": None, "error": True} for bid in biosample_ids}
    
    # Parse XML
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        print("Failed to parse XML.", file=sys.stderr)
        return {bid: {"country": None, "error": True} for bid in biosample_ids}
    
    results = {}
    for biosample in root.findall(".//BioSample"):
        acc = biosample.get("accession")
        if not acc:
            continue
        country = None
        for attr in biosample.findall(".//Attribute"):
            attr_name = attr.get("attribute_name")
            if attr_name and attr_name.lower() in [name.lower() for name in COUNTRY_ATTR_NAMES]:
                country = attr.text
                break
        results[acc] = {"country": country, "error": False}
    
    # Fill missing
    for bid in biosample_ids:
        if bid not in results:
            results[bid] = {"country": None, "error": True}
    return results

def main():
    # 1. Read run IDs
    print(f"Reading run IDs from {RUN_FILE}...")
    try:
        with open(RUN_FILE, 'r') as f:
            run_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File {RUN_FILE} not found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(run_ids):,} run IDs.")

    # 2. Read metadata CSV
    print(f"Reading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"File {CSV_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    # 3. Filter to only the runs we need
    if "run_accession" in df.columns:
        run_col = "run_accession"
    elif "Run" in df.columns:
        run_col = "Run"
    else:
        print("Could not find a column for run accession. Available columns:", df.columns.tolist(), file=sys.stderr)
        sys.exit(1)

    df_filtered = df[df[run_col].isin(run_ids)]
    print(f"Kept {len(df_filtered):,} rows matching the given run IDs.")

    if len(df_filtered) == 0:
        print("No matching runs found. Check column name and run IDs.", file=sys.stderr)
        sys.exit(1)

    # 4. Determine BioSample column
    if "biosample" in df_filtered.columns:
        id_col = "biosample"
        print("Using 'biosample' column for BioSample accessions.")
    elif "sample_accession" in df_filtered.columns:
        id_col = "sample_accession"
        print("Using 'sample_accession' column (may be less reliable).")
    else:
        print("Neither 'biosample' nor 'sample_accession' column found.", file=sys.stderr)
        sys.exit(1)

    biosample_list = df_filtered[id_col].dropna().unique().tolist()
    total_samples = len(biosample_list)
    print(f"Found {total_samples:,} unique BioSample accessions among filtered runs.")

    if total_samples == 0:
        print("No BioSample accessions found. Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # 5. Fetch country for each BioSample in batches (POST)
    country_dict = {}
    total_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Fetching metadata in batches of {BATCH_SIZE} (total {total_batches} batches)...")

    for i in tqdm(range(0, total_samples, BATCH_SIZE), desc="Fetching", unit="batch"):
        batch = biosample_list[i:i+BATCH_SIZE]
        batch_results = fetch_biosample_metadata(batch)
        country_dict.update(batch_results)
        time.sleep(SLEEP_TIME)

    # 6. Map results back to dataframe
    def get_country(bs_id):
        if pd.isna(bs_id):
            return None
        bs_id = str(bs_id).strip()
        return country_dict.get(bs_id, {}).get("country")
    
    def get_error(bs_id):
        if pd.isna(bs_id):
            return False
        bs_id = str(bs_id).strip()
        return country_dict.get(bs_id, {}).get("error", False)

    df_filtered["country"] = df_filtered[id_col].apply(get_country)
    df_filtered["country_error"] = df_filtered[id_col].apply(get_error)
    df_filtered["is_africa"] = df_filtered["country"].apply(is_african)

    total_runs = len(df_filtered)
    runs_with_country = df_filtered["country"].notna().sum()
    runs_african = df_filtered["is_africa"].sum()
    runs_non_african = (df_filtered["is_africa"] == False) & (df_filtered["country"].notna())
    runs_non_african_count = runs_non_african.sum()
    unknown = runs_with_country - runs_african - runs_non_african_count
    missing_country = total_runs - runs_with_country
    errors = df_filtered["country_error"].sum()

    print("\n" + "="*50)
    print("COUNTRY ORIGIN SUMMARY (filtered runs)")
    print("="*50)
    print(f"Total runs processed             : {total_runs:,}")
    print(f"Runs with country information    : {runs_with_country:,} ({runs_with_country/total_runs*100:.1f}%)")
    print(f"  - African                      : {runs_african:,} ({runs_african/total_runs*100:.1f}%)")
    print(f"  - Non-African                  : {runs_non_african_count:,} ({runs_non_african_count/total_runs*100:.1f}%)")
    print(f"  - Unknown (country present but not recognised) : {unknown:,}")
    print(f"Runs with missing country        : {missing_country:,} ({missing_country/total_runs*100:.1f}%)")
    print(f"  - of which fetch errors        : {errors:,}")
    print("="*50)

    # African country breakdown
    african_countries = df_filtered[df_filtered["is_africa"]]["country"].value_counts()
    if not african_countries.empty:
        print("\nAfrican country distribution (top 10):")
        print(african_countries.head(10).to_string())
    else:
        print("\nNo African countries found among the filtered runs.")

    # Show any remaining failed BioSample IDs
    failed_biosamples = df_filtered[df_filtered["country_error"]][id_col].dropna().unique()[:20]
    if len(failed_biosamples) > 0:
        print("\nFirst 20 BioSample IDs that still failed to fetch:")
        for bs in failed_biosamples:
            print(f"  {bs}")
        print("You can manually check these at https://www.ncbi.nlm.nih.gov/biosample/?term=<ID>")
    else:
        print("\nAll BioSample IDs fetched successfully!")

    # 8. Optionally save
    save = input("\nSave annotated CSV with country column? (y/n): ").strip().lower()
    if save == 'y':
        df_filtered.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()