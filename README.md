# Spark LLM-Agentic Mental Health Monitor

[![PySpark](https://img.shields.io/badge/PySpark-3.5.0-E25A1C?style=flat-square&logo=apachespark)](https://spark.apache.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Gemini](https://img.shields.io/badge/Gemini_3_Flash-4285F4?style=flat-square&logo=google)](https://deepmind.google/technologies/gemini/)

A big data analytics framework for screening mental health disorders — specifically **Depression** and **Anorexia** — by analyzing Reddit posts. The system uses **Apache Spark** for distributed processing over 600,000+ records and integrates an **agentic LLM** (Google Gemini 3 Flash) for dynamic, natural-language-driven data exploration.

> [!NOTE]
> This project was developed as part of an M.Sc. in Artificial Intelligence and Machine Learning at the University of Calabria.

---

## Architecture

The system is composed of four layers:

| Layer | Technology | Role |
|---|---|---|
| **Data Processing** | Apache Spark 3.5 (SQL, MLlib, Streaming) | Distributed ETL, aggregation, feature engineering |
| **Machine Learning** | PySpark ML Pipeline (TF-IDF + Logistic Regression) | Multi-class classification (Control / Depression / Anorexia) |
| **Agentic LLM** | Google Gemini 3 Flash | Natural-language code generation, topic labeling, ad-hoc analysis |
| **Dashboard** | Streamlit + Plotly Express | Interactive visualization, real-time monitoring, chat interface |

### Data Flow

```
eRisk 2018 XML Dataset
        │
        ▼
  make_csv_dataset2.py    ────  ETL: XML → CSV with label mapping
        │
        ▼
  training.py             ────  Train TF-IDF + Logistic Regression pipeline
        │                              │
        ▼                              ▼
  Streamlit App (app.py)        PySpark PipelineModel
        │                              │
        ├── Home — class balance, self-reference ratio, user ranking
        ├── Temporal — circadian rhythm, post volume over time
        ├── Content Search — keyword matching across 600K+ posts
        ├── AI & ML — feature importance, LDA topic modeling, LLM chat
        └── Streaming — real-time class distribution from live feed
                                  │
                                  ▼
  stream_simulator.py     ────  Simulates batch writes to test streaming
```

---

## Features

**Distributed Data Engine** — Spark Master-Worker topology processes 600K+ records with horizontal scalability.

**Psycholinguistic Analysis** — Computes the *Self-Reference Ratio* (first-person pronoun frequency), a clinical marker for mood disorders.

**Circadian Rhythm Analytics** — Normalizes post frequency across 24 hours to detect sleep-wake cycle anomalies associated with depression.

**Topic Modeling** — Unsupervised LDA clustering with LLM-generated semantic labels for each discovered topic.

**Agentic LLM Interface** — A chat assistant that translates natural-language questions into executable PySpark code, enabling ad-hoc analysis without writing queries.

**Feature Importance Visualization** — Inspects Logistic Regression coefficients to understand which terms drive each classification.

**Streaming Analytics** — Spark Structured Streaming monitors live data batches, updating class distribution metrics in real time.

---

## Getting Started

### Prerequisites

- Java 11+ (required by PySpark)
- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API credentials

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

### 3. Prepare the dataset

Convert the eRisk 2018 XML corpus to a Spark-optimized CSV:

```bash
python make_csv_dataset2.py
```

> [!CAUTION]
> The ETL script contains hardcoded absolute paths. Edit `make_csv_dataset2.py` to point `path_dep`, `path_ano`, and `output_file` to your local eRisk 2018 dataset location.

### 4. Train the ML pipeline

Train the Logistic Regression classifier and save the `PipelineModel`:

```bash
python training.py
```

This produces a `models/lr_pipeline_model/` directory containing the trained pipeline with TF-IDF vectorizer and classifier.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard will initialize a SparkSession and load the data into memory.

### 6. (Optional) Simulate streaming data

In a separate terminal, run the batch simulator to test real-time analytics:

```bash
python stream_simulator.py
```

This writes randomized batches of 20 rows every 3 seconds to a `stream_input/` directory, which Spark Structured Streaming picks up automatically.

---

## Screenshots

| Home Dashboard | Feature Importance |
|---|---|---|
| ![Home](https://github.com/user-attachments/assets/e0d3f7f7-cce2-46c1-a853-1682db903eee) | ![Feature Importance](https://github.com/user-attachments/assets/d82bc58c-40ef-4bd0-acbf-434b3d95e468) |

| LLM Chat | LLM Chat | LLM Chat |
|---|---|---|
| ![Chat1](https://github.com/user-attachments/assets/b76b26d6-00ac-4a05-bf32-a7dc3ed3734c) | ![Chat2](https://github.com/user-attachments/assets/c594b016-b100-491a-be2d-1b85d6a40f54) | ![Chat3](https://github.com/user-attachments/assets/015973cc-afe4-4b6a-a98a-344d4caf9149) |

---

## Project Structure

```
├── app.py                  # Streamlit dashboard (6 pages)
├── logica.py               # Spark data processing logic and analytics
├── AI_ML_LLM.py            # LLM agent, LDA topic modeling, feature extraction
├── training.py             # ML pipeline training script
├── make_csv_dataset2.py    # XML-to-CSV ETL for eRisk 2018
├── stream_simulator.py     # Streaming data simulator
├── requirements.txt        # Python dependencies
└── models/                 # Trained PipelineModel (generated by training.py)
```

---

## Dataset

This project uses the [eRisk 2018](https://erisk.irlab.org/) dataset, which contains Reddit posts from users labeled as:
- **0** — Control (healthy)
- **1** — Depression
- **2** — Anorexia

The dataset is **not** included in this repository and must be obtained from the eRisk organizers.

---

## Author

**Luca Timpano**  
M.Sc. Artificial Intelligence and Machine Learning  
University of Calabria
