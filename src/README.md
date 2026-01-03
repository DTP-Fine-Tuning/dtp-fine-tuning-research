# src directory

`src/` directory merupakan direktori yang digunakan untuk menampung berbagai skrip python untuk menjalankan berbagai tugas fine-tuning SFT. Direktori tersebut terdiri atas 3 directory utama yakni `eval/`, `inference/`, `training/`. Tiga direktori tersebut akan mempermudah proses fine-tuning dengan otomasi shell script pada direktori `scripts` untuk menjalankan pipeline yang lebih optimal.

## Quick Overview
### `eval/`
Direktori `eval/` digunakan untuk melakukan evaluasi model yang telah dilakukan proses fine tuning guna melakukan validasi model dengan berbagai pendekatan evaluasi. Library eval yang digunakan pada direktori `eval/` adalah deepeval yang mendukung single turn evaluation dan multi-turn evaluation sesuai dengan use case model Digital Talent Pool.

### `inference/`
Direktori `inference/` digunakan untuk melakukan proses inferencing model dari hasil fine tuning yang telah dilakukan sebagai langkah uji coba penerapan human evaluation berdasarkan data tes yang dimiliki oleh tim DTP 2. Library yang digunakan pada direktori `inference/` adalah `gradio` dan `bitsandbytes` untuk melakukan proses inferencing secara interaktif oleh tim DTP 2.

### `training/`
Direktori `training/` digunakan untuk melakukan proses fine tuning model dengan teknik Supervised Fine Tuning (SFT). Tim DTP 2 melakukan proses SFT menggunakan beberapa library pendukung seperti **unsloth**, **trl** dan **transformers** sesuai dengan kapasitas GPU yang dimiliki agar menghasilkan model SFT yang robust.

## Get in Touch with Maintainers
### Wildan: [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/wildanaziz) | [![Firefox](https://img.shields.io/badge/Firefox-FF7139?logo=firefoxbrowser&logoColor=white)](https://wildanaziz.vercel.app/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/wildanaziz)
### Syafiq: [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/syafiqirz)
### Naufal: [![GitHub](https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white)](https://github.com/NaufalArsa)

## Special Thanks to
1. **[TRL Official](https://huggingface.co/docs/trl/sft_trainer)**
2. **[SMOL Course](https://huggingface.co/learn/smol-course/unit0/1)**
3. **[Unsloth Official](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)**
4. **[Gradio](https://www.gradio.app/docs)**
