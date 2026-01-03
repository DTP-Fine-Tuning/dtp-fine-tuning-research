# dtp-fine-tuning-research

Repositori ini berfokus pada **riset dan eksperimen fine-tuning LLM** menggunakan pendekatan SFT (Supervised Fine-Tuning)

##  Tujuan

* Menyediakan kerangka eksperimen untuk fine-tuning LLM.
* Mendukung reproducibility dengan konfigurasi terstruktur.
* Menghubungkan hasil preprocessing dari repo [`dtp-data-pipeline`](https://github.com/DTP-Fine-Tuning/dtp-data-pipeline).

##  Struktur Direktori

```
dtp-fine-tuning-research/
├── configs/                # Konfigurasi eksperimen (YAML)
├── experiments/            # Log & hasil eksperimen
├── notebooks/              # Exploratory Jupyter notebooks
├── scripts/                # Script eksekusi training
├── src/                    # Source code (trainer, utils, dsb.)
├── paper/                  # knowledge base fine tuning
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
* [TRL](https://huggingface.co/docs/trl/index) - untuk sft
* [Weights & Biases](https://wandb.ai/) – tracking eksperimen
* [Unsloth](https://unsloth.ai/docs) – untuk faster sft

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

* [Custom Scenarios Guidelines](docs/custom_scenarios_guide.md)
* [Evaluation Guidelines](docs/eval_guide.md)
* [Loss Curve Guidelines](docs/loss_curve_guide.md)
* [Scripts Installation Guidelines](docs/scripts_installation_guide.md)

## How to Contribute?

Anda dapat melakukan forking repository `dtp-fine-tuning-research` dan melakukan pull request.

---

 *Repo ini saling melengkapi dengan [`dtp-data-pipeline`](https://github.com/DTP-Fine-Tuning/dtp-data-pipeline) sebagai sumber dataset preprocessing.*
