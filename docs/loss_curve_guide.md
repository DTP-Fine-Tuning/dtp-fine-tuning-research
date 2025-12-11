# PANDUAN STRATEGI & ANALISIS KURVA LOSS PER FASE
**Konteks:** Fine-Tuning LLM (Qwen3) | **Metode:** SFT (Full/LoRA)

---

## FASE 1: AWAL TRAINING (Steps 0 - 100)
**Fokus:** Adaptasi Format & Stabilitas Gradien.
Pada fase ini, `warmup_ratio` (misal 0.05) sedang bekerja menaikkan Learning Rate dari 0 ke target (misal 2e-5).

### 1. Pola: "The Sharp Drop" (Penurunan Tajam)
Loss turun drastis (misal dari 3.0 ke 0.8) dalam waktu singkat.
* **Diagnosa A (Positif - Wajar):** Model sedang belajar **Format/Sintaks**.
    * *Penjelasan:* Model sudah pintar bahasa, tapi baru belajar format dataset Anda (misal: "Oh, saya harus pakai JSON", atau "Oh, saya harus jawab singkat").
    * *Tindakan:* **Lanjutkan.** Cek log evaluasi pertama.
* **Diagnosa B (Negatif - Data Leakage):** Loss jatuh ke dekat 0 (misal 0.05) dalam <20 steps.
    * *Penjelasan:* Model "mencontek". Jawaban (labels) bocor ke dalam pertanyaan (input).
    * *Tindakan:* **STOP SEGERA.** Periksa dataset & DataCollator Anda.

### 2. Pola: "The Spike" (Lonjakan)
Loss turun, lalu tiba-tiba melonjak naik tinggi, atau menjadi `NaN`.
* **Diagnosa:** Gradient Explosion. Learning Rate terlalu tinggi atau Warmup terlalu cepat.
* **Tindakan:**
    * Restart training.
    * Naikkan `warmup_ratio` (misal ke 0.1).
    * Turunkan Learning Rate.

---

## FASE 2: PERTENGAHAN (Cruising Phase)
**Fokus:** Penyerapan Pengetahuan (Knowledge Absorption).
Fase dimana Learning Rate stabil di angka maksimal atau mulai turun perlahan (decay).

### 1. Pola: Penurunan Konsisten
Training Loss turun pelan, Validation Loss turun pelan.
* **Status:** **Sehat.**
* **Strategi:** Jangan sentuh apa pun. Biarkan model belajar.

### 2. Pola: Fluktuasi Tinggi (Noise)
Grafik naik-turun kasar seperti gergaji.
* **Diagnosa:** Batch Size terlalu kecil.
* **Strategi:**
    * Tidak perlu dihentikan jika tren rata-rata masih turun.
    * Untuk next training: Naikkan `gradient_accumulation_steps`.

---

## FASE 3: AKHIR (Steps 400+ / Convergence)
**Fokus:** Generalisasi vs Overfitting.
Disini keputusan "Stop or Continue" sangat krusial.

### 1. Pola: "The Plateau" (Datar / Osilasi)
Loss curve bergerak naik-turun dengan selisih sangat tipis (misal 0.001 - 0.005) antar evaluasi.
* **Analisis:** *Diminishing Returns*. Model sudah jenuh. Melanjutkan training hanya membuang komputasi.
* **Tindakan:**
    * Cek **Validation Loss**. Jika datar 2-3x berturut-turut, **STOP TRAINING**.
    * Gunakan checkpoint terbaik (bukan terakhir).

### 2. Pola: "Overfitting" (Simpang Jalan)
Training Loss terus turun tajam, TAPI Validation Loss mulai **NAIK**.
* **Analisis:** Model menghafal data training, tapi bodoh saat dites data baru.
* **Tindakan:**
    * **STOP TRAINING SEKARANG.**
    * Model di step-step sebelumnya (saat Val Loss terendah) adalah model terbaik Anda.

---

## CHEATSHEET STRATEGI PENCEGAHAN

| Gejala Grafik | Fase Kemunculan | Penyebab Utama | Solusi Langsung |
| :--- | :--- | :--- | :--- |
| **Loss ~ 0.0 instan** | Awal (0-50) | Data Leakage (Bocor) | Stop, Cek Dataset & Masking |
| **Loss Spike / NaN** | Awal / Tengah | LR ketinggian / Data kotor | Turunkan LR, Tambah Warmup |
| **Loss Datar (High)** | Awal / Tengah | LR kekecilan / Model macet | Naikkan LR (Hati-hati) |
| **Val Loss Naik** | Akhir (400+) | Overfitting | Stop, Load Best Checkpoint |
| **Val Loss Datar** | Akhir (400+) | Converged (Selesai) | Stop, Training Selesai |

### Wajib tahu:
> **"Checkpoint Terbaik Bukanlah Checkpoint Terakhir."**