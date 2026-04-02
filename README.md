# Violin: Volumetric Injection Attacks Against Searchable Encryption

`Violin` is the reference research code for reproducing the experiments in the paper _Violin: Powerful Volumetric Injection Attack Against Searchable Encryption With Optimal Injection Size_. The repository evaluates the proposed `Violin` attack and compares it against baselines such as `Decoding`, `BVA`, and `BVMA` under multiple datasets and defense settings.

This project is useful for researchers working on:

- searchable symmetric encryption (SSE)
- leakage-abuse and injection attacks
- volumetric leakage analysis
- defense evaluation for `SEAL`, `ShieldDB`, and update scenarios

## Highlights

- Reproduces the main experimental settings from the paper
- Includes attack implementations for standard, update-aware, `SEAL`, and `ShieldDB` settings
- Covers multiple evaluation dimensions: recovery rate, injected volume, injected file count, and runtime
- Ships with processed dataset artifacts in `Datasets/` for easier experimentation

## Repository Layout

- `attacks/`: core attack implementations for the standard setting
- `attacks_Update/`: attack implementations for active client update scenarios
- `attacks_seal/`: attack implementations for static padding / `SEAL`
- `attacks_ShieldDB/`: attack implementations for dynamic padding / `ShieldDB`
- `selectivity_*.py`: evaluates the impact of query selectivity on recovery rate
- `m_enron.py`: evaluates the impact of the number of queries on recovery rate and injection overhead
- `n_*.py`: evaluates the impact of the keyword set size on recovery rate, injection volume, and runtime
- `padding_SEAL_enron.py`: evaluates static padding with `SEAL`
- `padding_ShieldDB_*.py`: evaluates dynamic padding with `ShieldDB`
- `ThresholdCounter.py`: evaluates injection volume under threshold-based countermeasures
- `Update.py`: evaluates recovery rate when the client actively updates the dataset
- `injectionPadding.py`: evaluates different injection padding parameters
- `Datasets/`: processed pickle files used by the experiments

## Installation

Use Python 3.9+ and install dependencies with:

```bash
pip install -r requirements.txt
```

The scripts write cached results to `pkl/` and figures to `pic/`. Create both directories before running experiments:

```bash
mkdir -p pkl pic
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force pkl, pic
```

## Quick Start

Run any experiment script directly. Examples:

```bash
python selectivity_Enron.py
python n_enron.py
python padding_SEAL_enron.py
python Update.py
```

Most scripts follow the same workflow:

- load processed dataset files from `Datasets/`
- run repeated attack simulations
- save intermediate results to `pkl/`
- export plots to `pic/`

## Datasets

The paper uses the following public datasets:

- Enron: https://www.cs.cmu.edu/~enron/
- Lucene: http://mail-archives.apache.org/mod_mbox/lucene-java-user
- Movies: http://www.cs.cmu.edu/~ark/personas/

This repository already includes processed dataset artifacts such as tokenized metadata and keyword-volume mappings in `Datasets/`.

Common processed files include:

- `doc_size_*.pkl`: document size information for each dataset
- `*_vol_access.pkl`: mappings from keywords to the sizes of matching documents

## Reproducibility Notes

- The code is organized as experiment scripts rather than a packaged library
- Several experiments generate publication-style figures using `matplotlib` and `seaborn`
- Some plots use LaTeX text rendering; if figure generation fails on your machine, verify that your local plotting/LaTeX environment is available

## Citation

If you use this repository in academic work, please cite:

```bibtex
@article{ZhangWWWS25,
  author       = {Lei Zhang and
                  Jianfeng Wang and
                  Jiaojiao Wu and
                  Yunling Wang and
                  Shifeng Sun},
  title        = {Violin: Powerful Volumetric Injection Attack Against Searchable Encryption
                  With Optimal Injection Size},
  journal      = {{IEEE} Trans. Dependable Secur. Comput.},
  volume       = {22},
  number       = {4},
  pages        = {4103--4115},
  year         = {2025},
  url          = {https://doi.org/10.1109/TDSC.2025.3543248},
  doi          = {10.1109/TDSC.2025.3543248},
  timestamp    = {Sat, 09 Aug 2025 01:00:00 +0200}
}
```

## License

See `LICENSE` for licensing information.
