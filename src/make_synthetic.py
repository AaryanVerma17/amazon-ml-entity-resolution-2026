"""Generates a small fake dataset matching the exact challenge schema, so the
pipeline can be smoke-tested before the real dataset drops.
"""
import random
import pandas as pd
import os

random.seed(0)

NAMES = ["Reliance Fresh", "McDonald's", "Walmart", "Starbucks", "Reliance Digital",
          "Cafe Coffee Day", "Big Bazaar", "Domino's Pizza", "Pizza Hut", "Subway"]
SUFFIXES = ["", " Pvt Ltd", " Private Limited", " Corp", " & Co"]
CITIES = [("24 MG Road", "Bangalore", "560001", "India"),
          ("100 Main St", "Austin", "73301", "US"),
          ("5 Rue de Paris", "Paris", "75001", "France"),
          ("Sector 18", "Delhi", "110001", "India")]


def noisy_name(name):
    variants = [name, name.replace("MG", "Mahatma Gandhi"), name.upper(), name.lower()]
    return random.choice(variants) + random.choice(SUFFIXES)


def noisy_address(addr, city, postal, add_postal=True):
    forms = [f"{addr}, {city}", f"{addr} {city}", f"Near {city} Metro, {addr}"]
    a = random.choice(forms)
    return f"{a} {postal}" if add_postal and random.random() > 0.3 else a


def gen(n_entities, out_dir):
    os.makedirs(f"{out_dir}/train", exist_ok=True)
    os.makedirs(f"{out_dir}/test", exist_ok=True)

    for split, offset in (("train", 0), ("test", 10000)):
        s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
        for i in range(n_entities):
            eid = offset + i
            name = random.choice(NAMES)
            addr, city, postal, country = random.choice(CITIES)
            s1_id = f"S1-{eid:05d}"
            s1_rows.append({"entity_id": s1_id, "business_name": name,
                             "business_address": f"{addr}, {city} {postal}", "country": country})

            matches = []
            n_match = random.choices([0, 1, 2], weights=[0.3, 0.5, 0.2])[0]
            for k in range(n_match):
                target = s2_rows if k % 2 == 0 else s3_rows
                mid = f"{'S2' if target is s2_rows else 'S3'}-{eid:05d}{k}"
                target.append({"entity_id": mid, "business_name": noisy_name(name),
                                "business_address": noisy_address(addr, city, postal), "country": country})
                matches.append(mid)

            # noise: unrelated records in S2/S3 (non-matches / distractors)
            if random.random() > 0.5:
                junk_name = random.choice(NAMES)
                jaddr, jcity, jpostal, jcountry = random.choice(CITIES)
                s2_rows.append({"entity_id": f"S2-J{eid:05d}", "business_name": noisy_name(junk_name),
                                 "business_address": noisy_address(jaddr, jcity, jpostal), "country": jcountry})

            if split == "train":
                gt_rows.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matches)})

        pd.DataFrame(s1_rows).to_csv(f"{out_dir}/{split}/{split}_source1.tsv", sep="\t", index=False)
        pd.DataFrame(s2_rows).to_csv(f"{out_dir}/{split}/{split}_source2.tsv", sep="\t", index=False)
        pd.DataFrame(s3_rows).to_csv(f"{out_dir}/{split}/{split}_source3.tsv", sep="\t", index=False)
        if split == "train":
            pd.DataFrame(gt_rows).to_csv(f"{out_dir}/train/train_ground_truth.tsv", sep="\t", index=False)


if __name__ == "__main__":
    gen(300, "dataset")
    print("synthetic dataset written to dataset/")
