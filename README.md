# Spark-LLM-Agentic-Mental-Health-Monitor

This repository contains a big data analytics framework for screening mental health disorders, specifically Depression and Anorexia, by analyzing Reddit posts. The system leverages **Apache Spark** for distributed processing and integrates an **agentic LLM interface** using Google’s **gemini-3-flash-preview** for dynamic data exploration.

---

## Technical Architecture

The system is designed with a modular architecture to separate data processing logic, machine learning pipelines, and the user interface.

### 1. Agentic LLM Implementation

The integration of the `gemini-3-flash-preview` model allows the system to function as an agent capable of interpreting natural language and interacting with the Spark environment.

* 
**Code Generation**: The LLM translates user prompts into executable PySpark code.


* 
**Dynamic Execution**: Generated code is executed at runtime within the active `SparkSession` context using Python's `exec()` function.


* 
**Semantic Labeling**: The agent interprets raw clusters from Latent Dirichlet Allocation (LDA) to provide human-readable thematic titles.



### 2. Distributed Data Processing

* 
**Spark Engine**: Handles over 600,000 records using a Master-Worker topology for horizontal scalability.


* 
**Psycholinguistic Analysis**: Calculates the *Self-Reference Ratio* (I-usage frequency) as a clinical marker for mood disorders.


* 
**Temporal Analytics**: Examines circadian rhythms by normalizing post frequency across 24 hours to identify sleep-wake cycle anomalies.


* 
**Structured Streaming**: Monitors live data batches to update class distribution metrics in real-time.



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

**Author**: Luca Timpano

**Course**: M.Sc. in Artificial Intelligence and Machine Learning 

**Institution**: University of Calabria 
