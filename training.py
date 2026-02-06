import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, RegexTokenizer, StopWordsRemover, CountVectorizer, IDF

# --- CONFIGURAZIONE PERCORSI RELATIVI ---
CSV_FILE = "all_user_with_anorexia.csv"
MODEL_REL_PATH = "models/lr_pipeline_model"

def run_training():
    # Inizializzazione sessione Spark con allocazione memoria dedicata
    spark = SparkSession.builder \
        .appName("Training_eRisk_Model") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    print(f"Caricamento dati da: {CSV_FILE}")
    
    if not os.path.exists(CSV_FILE):
        print(f"Errore: Il file {CSV_FILE} non e stato trovato.")
        spark.stop()
        return

    # Caricamento del dataset
    df = spark.read \
        .option("header", "true") \
        .option("multiLine", "true") \
        .option("sep", ",") \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .csv(CSV_FILE)

    # Preprocessing dei tipi e pulizia record nulli
    df = df.withColumn("label", col("label").cast(IntegerType())) \
           .filter(col("label").isNotNull()) \
           .fillna("")

    # mappa le classi (Control, Depression, Anorexia)
    indexer = StringIndexer(inputCol="label", outputCol="label_index", handleInvalid="skip")
    
    # Text Preprocessing
    tokenizer = RegexTokenizer(
        inputCol="text", 
        outputCol="raw_words", 
        pattern="[a-zA-Z]{3,}", 
        gaps=False
    )
    
    remover = StopWordsRemover(inputCol="raw_words", outputCol="filtered")
    custom_stops = remover.getStopWords() + [
        "like", "get", "one", "think", "really", "know", "even", 
        "actually", "also", "good", "well", "would", "could", "haha", "yea", "yup"
    ]   
    remover.setStopWords(custom_stops)
    
    # Vettorizzazione TF-IDF
    cv = CountVectorizer(inputCol="filtered", outputCol="raw_features", vocabSize=2000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # Modello Lineare (Logistic Regression)
    lr = LogisticRegression(labelCol="label_index", featuresCol="features", regParam=0.05)
    
    # Definizione ed esecuzione della Pipeline
    pipeline = Pipeline(stages=[indexer, tokenizer, remover, cv, idf, lr])
    
    print("Addestramento del modello in corso...")
    model = pipeline.fit(df)
    
    # Salvataggio persistente del modello (PipelineModel)
    print(f"Salvataggio modello in: {MODEL_REL_PATH}")
    model.write().overwrite().save(MODEL_REL_PATH)
    
    print("Addestramento e salvataggio completati con successo.")
    spark.stop()

if __name__ == "__main__":
    run_training()