"""Meaning of the integer labels in each dataset file and how the loader maps them.

`raw` is the meaning of each integer in the .npy file; `verified` records whether that meaning was checked against an
external phenotype table. Binary metrics treat the loaded label 1 as the positive class (sensitivity = recall of label 1).
Only ADHD-200 is flipped on load, so that label 1 = patient for every binary dataset.

  abide  1 = ASD, 0 = typical control; subject ids and sex were checked against the public ABIDE-I phenotypic table.
  adni   Not verified externally (the phenotype table requires registered access). Label 0 was acquired almost entirely
         with one protocol (TR 0.607 s, 482 volumes), while label 1 spans four protocols; see WARNINGS.
  ppmi   0 control, 1 PD, 2 SWEDD, 3 prodromal, as distributed with the preprocessed data; not verified externally.
  adhd   As documented by the data provider (label 1 = control, 0 = ADHD); the file carries no subject ids or phenotype.
"""

LABELS = {
    "abide": {"raw": {0: "TDC", 1: "ASD"}, "verified": True, "loader_flips": False,
              "sex": {0: "male", 1: "female"}},
    "adni": {"raw": {0: "unknown (CN or AD)", 1: "unknown (AD or CN)"}, "verified": False, "loader_flips": False,
             "sex": {0: "unknown", 1: "unknown"}},
    "ppmi": {"raw": {0: "control (assumed)", 1: "PD (assumed)", 2: "SWEDD (assumed)", 3: "prodromal (assumed)"},
             "verified": False, "loader_flips": False, "sex": {0: "male (inferred)", 1: "female (inferred)"}},
    "adhd": {"raw": {0: "ADHD", 1: "control"}, "verified": False, "documented": True, "loader_flips": True,
             "sex": None},
}


def loaded_label_names(family):
    """{loaded label: name} after common.load_dataset (which flips ADHD-200 only)."""
    info = LABELS[family.lower()]
    raw = info["raw"]
    if info["loader_flips"]:
        return {1 - k: v for k, v in raw.items()}
    return dict(raw)


def positive_class_name(family):
    """What 'sensitivity' (recall of loaded label 1) is the recall of, for a binary dataset."""
    return loaded_label_names(family)[1]


# Known properties of the data that affect how results should be read (printed at the start of every run).
WARNINGS = {
    "adni": ["acquisition confound: the label is largely predictable from scan protocol alone (TR + number of volumes, AUC 0.77); "
             "the 26 197-volume scans are all label 1"],
    "ppmi": ["very small classes (15 / 113 / 14 / 53 subjects): metrics on one split are noisy"],
    "adhd": ["54 exact duplicate scans (kept, placed in one partition); no subject ids, so same-person rescans cannot be detected",
             "label meaning (1 = control, 0 = patient) is as documented by the data owner; the file itself cannot confirm it"],
}


def data_warnings(family):
    return list(WARNINGS.get(family.lower(), []))
