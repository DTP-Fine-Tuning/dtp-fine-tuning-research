# dtp-fine-tuning-research

Repositori ini berfokus pada **riset dan eksperimen fine-tuning LLM** menggunakan pendekatan SFT (Supervised Fine-Tuning) dan DPO (Direct Preference Optimization).

##  Tujuan

* Menyediakan kerangka eksperimen untuk fine-tuning LLM.
* Mendukung reproducibility dengan konfigurasi terstruktur.
* Menghubungkan hasil preprocessing dari repo [`llm-data-pipeline`](../llm-data-pipeline).

##  Struktur Direktori

```
dtp-fine-tuning-research/
├── configs/                # Konfigurasi eksperimen (YAML)
├── experiments/            # Log & hasil eksperimen
├── notebooks/              # Exploratory Jupyter notebooks
├── scripts/                # Script eksekusi training
├── src/                    # Source code (trainer, utils, dsb.)
├── docs/                   # Dokumentasi riset
├── requirements.txt        # Dependensi utama
├── .gitignore
└── README.md
```

## Tools

* [Transformers](https://huggingface.co/transformers/) – model & training API
* [Datasets](https://huggingface.co/docs/datasets) – data handling
* [PEFT](https://github.com/huggingface/peft) – parameter-efficient tuning (LoRA, QLoRA)
* [Accelerate](https://huggingface.co/docs/accelerate) – distribusi training
* [TRL](https://huggingface.co/docs/trl/index) - untuk dft sft
* [Weights & Biases](https://wandb.ai/) – tracking eksperimen

##  Getting Started

1. Clone repo:

   ```bash
   git clone git@github.com:<your-org>/dtp-fine-tuning-research.git
   cd dtp-fine-tuning-research
   ```

2. Buat virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan eksperimen baseline:

   ```bash
   bash scripts/run_sft.sh
   ```

##  Dokumentasi

* [Fine-tuning Overview](docs/fine_tuning_overview.md)
* [Experiment Guidelines](docs/experiment_guidelines.md)

## How to Contribute?

Setiap anggota tim bisa menjalankan eksperimen dengan konfigurasi masing-masing. Gunakan direktori `experiments/` untuk menyimpan hasil & log.

---

 *Repo ini saling melengkapi dengan [`dtp-data-pipeline`](../dtp-data-pipeline) sebagai sumber dataset preprocessing.*
