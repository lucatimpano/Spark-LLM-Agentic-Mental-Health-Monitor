# Spark-LLM-Agentic-Mental-Health-Monitor

This repository contains a big data analytics framework for screening mental health disorders, specifically Depression and Anorexia, by analyzing Reddit posts. The system leverages **Apache Spark** for distributed processing and integrates an **agentic LLM interface** using Google’s **gemini-3-flash-preview** for dynamic data exploration.

---

## Technical Architecture

The system is designed with a modular architecture to separate data processing logic, machine learning pipelines, and the user interface.

### 1. Agentic LLM Implementation

The integration of the `gemini-3-flash-preview` model allows the system to function as an agent capable of interpreting natural language and interacting with the Spark environment.

* **Code Generation**: The LLM translates user prompts into executable PySpark code.


* **Dynamic Execution**: Generated code is executed at runtime within the active `SparkSession` context using Python's `exec()` function.


* **Semantic Labeling**: The agent interprets raw clusters from Latent Dirichlet Allocation (LDA) to provide human-readable thematic titles.



### 2. Distributed Data Processing

* **Spark Engine**: Handles over 600,000 records using a Master-Worker topology for horizontal scalability.


* **Psycholinguistic Analysis**: Calculates the *Self-Reference Ratio* (I-usage frequency) as a clinical marker for mood disorders.


* **Temporal Analytics**: Examines circadian rhythms by normalizing post frequency across 24 hours to identify sleep-wake cycle anomalies.


* **Structured Streaming**: Monitors live data batches to update class distribution metrics in real-time.



---

## Features Overview

**Data Engine** Apache Spark 3.5.0 (SQL, MLlib, Streaming) 

**Language Model** Google Gemini-3-Flash-Preview 

**Clinical Markers** Self-reference ratio, circadian rhythm analysis, and stylistic variance 

**Topic Modeling** Non-supervised clustering via LDA with LLM-based labeling 

**Visualization**  Interactive Streamlit dashboard with Plotly Express charts 

---

## Setup and Execution

### 1. Environment Configuration

Install the required dependencies, ensuring compatibility with your Spark and Java environment.

```bash
pip install -r requirements.txt

```

### 2. API Credentials

The agent requires a Google Gemini API key. Configure it in the Streamlit secrets file.

* **Path**: `.streamlit/secrets.toml`
* **Content**: `GEMINI_API_KEY = "YOUR_API_KEY_HERE"`

### 3. Data Preprocessing (ETL)

The eRisk 2018 dataset (XML) must be converted to a Spark-optimized CSV format.

```bash
python make_csv_dataset2.py

```

### 4. Machine Learning Training

Train the Logistic Regression pipeline to generate the `PipelineModel` and TF-IDF vocabulary.

```bash
python training.py

```

### 5. Launch the Application

Start the Streamlit dashboard to initialize the SparkSession and load data into memory.

```bash
streamlit run app.py

```

### 6. Streaming Simulation (Optional)

To test the real-time analytics module, run the batch simulator in a separate terminal.

```bash
python stream_simulator.py

```
---
### 7. Some Screenshot
<img width="1924" height="936" alt="screenshot_schermata_home" src="https://github.com/user-attachments/assets/e0d3f7f7-cce2-46c1-a853-1682db903eee" />
<img width="912" height="762" alt="feature_importance" src="https://github.com/user-attachments/assets/d82bc58c-40ef-4bd0-acbf-434b3d95e468" />
<img width="495" height="699" alt="chat1" src="https://github.com/user-attachments/assets/b76b26d6-00ac-4a05-bf32-a7dc3ed3734c" />
<img width="495" height="699" alt="chat2" src="https://github.com/user-attachments/assets/c594b016-b100-491a-be2d-1b85d6a40f54" />
<img width="495" height="699" alt="chat3" src="https://github.com/user-attachments/assets/015973cc-afe4-4b6a-a98a-344d4caf9149" />

---

**Author**: Luca Timpano

**Course**: M.Sc. in Artificial Intelligence and Machine Learning 

**Institution**: University of Calabria 
