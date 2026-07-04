import pandas as pd
from pysradb import SRAweb
import time

with open("data/SRR_Acc_List.txt", "r") as f:
    accession_ids = [line.strip() for line in f if line.strip()]

db = SRAweb()

print(f"fetching metadata for {len(accession_ids)} samples...")

BATCH_SIZE = 200 
metadata_frames = []

for i in range(0, len(accession_ids), BATCH_SIZE):
    batch = accession_ids[i:i + BATCH_SIZE]
    current_batch_num = (i // BATCH_SIZE) + 1
    total_batches = (len(accession_ids) // BATCH_SIZE) + 1
    
    print(f"processing batch {current_batch_num}/{total_batches}... (IDs {i} to {min(i + BATCH_SIZE, len(accession_ids))})")
    
    try:
        batch_df = db.sra_metadata(batch)
        if batch_df is not None and not batch_df.empty:
            metadata_frames.append(batch_df)
    except Exception as e:
        print(f"error!!! fetching batch starting at index {i}: {e}")
        print("Skipping this batch and continuing...")
    
    time.sleep(1)

if metadata_frames:
    metadata_df = pd.concat(metadata_frames, ignore_index=True)
    metadata_df.to_csv("sra_master_metadata.csv", index=False)
    print("\ndone! metadata saved to sra_master_metadata.csv")
else:
    print("\nfailed to retrieve any metadata")
