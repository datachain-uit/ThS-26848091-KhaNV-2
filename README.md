# ThS-26848091-KhaNV: Causal Machine Learning in Retail Marketing

**Author:** Nguyễn Viết Kha (`26848091`)  
**Institution:** University of Information Technology - VNU-HCM (UIT)  
**Research Topic:** *Selecting Adjustment Sets via Tiered Causal Graphs for Marketing Treatment Effect Estimation: An Evaluation Using Randomized Controlled Trials in Retail*

---

## 1. Repository Structure (Code-Only Public Repository)

```text
ThS-26848091-KhaNV/
├── configs/                        # Experiment configurations (YAML)
│   └── experiment_config.yaml
├── data/                           # Local dataset cache (gitignored)
│   ├── raw/
│   ├── processed/
│   └── osrct/
├── notebooks/                      # Executed Jupyter Notebooks with reproducible outputs
│   └── 00_hillstrom_eda_and_rct_ate.ipynb
├── results/                        # Generated tables and figures
│   ├── raw/
│   ├── tables/
│   └── figures/
├── scripts/                        # Automation & synchronization scripts
│   ├── sync_drive.py               # Sync deliverables with Google Drive working folder
│   ├── generate_weekly_email.py    # Compile weekly worklog to HTML/.eml and Gmail Drafts
│   └── fetch_advisor_email.py      # Fetch advisor emails via IMAP
├── src/                            # Core 5-step Causal ML Pipeline
│   ├── data/
│   │   ├── datasets.py             # Dataset loaders & 3-tier variable partitioning
│   │   └── osrct.py                # Pool-anchored OSRCT biased sampling (Gentzel et al., 2021)
│   ├── discovery/
│   │   └── tiered_discovery.py     # Expert DAG & Temporal Tier constraints (K0-K3)
│   ├── adjustment/
│   │   └── strategies.py           # Exact O-set (Henckel et al., 2022), OAL (Shortreed & Ertefaie, 2017), S0-S6
│   ├── estimation/
│   │   └── estimators.py           # Cross-fitted AIPW / Doubly Robust & DML / Causal Forest
│   ├── evaluation/
│   │   └── metrics.py              # |Bias|, MCSE (Morris et al., 2019), RMSE, Normalized AUUC, Qini, Wilcoxon + Holm
│   └── utils/
│       └── reproducibility.py      # Global seed, YAML config loader, MD5 checksums
├── tests/                          # Unit tests (O-set, OAL, OSRCT pool ground truth, Placebo check)
│   └── test_osrct_sanity.py
└── requirements.txt                # Pinned Python dependencies
```

---

## 2. Quickstart & Verification

```powershell
# 1. Run unit tests (verifies exact O-set, Outcome-Adaptive Lasso, and pool-anchored OSRCT)
.\.venv\Scripts\python.exe -m pytest tests/test_osrct_sanity.py -v
```
