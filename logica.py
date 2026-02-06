# logica.py
import os
import pandas as pd

# Importazioni PySpark Core & SQL
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, try_to_timestamp, lit, variance, length, sum, desc, avg, 
    rand, hour, lower, size, split, to_date, count
)
from pyspark.sql.types import IntegerType, StructType

from AI_ML_LLM import AI

class LogicaReddit:
    def __init__(self, file_path):
        """
        Inizializza la sessione Spark e configura il processore dati.
        """
        master_url = os.getenv("SPARK_MASTER_URL", "local[*]")
        
        self.spark = SparkSession.builder \
            .appName("eRisk_Processor") \
            .master(master_url) \
            .config("spark.executor.memory", "1g") \
            .getOrCreate()
            
        self.path = file_path
        self.df = None
        self.ai = AI()

    def load_data(self):
        """
        Carica il CSV, gestisce il casting dei tipi e pulisce i nulli.
        """
        self.df = self.spark.read \
            .option("header", "true") \
            .option("multiLine", "true") \
            .option("sep", ",") \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("ignoreLeadingWhiteSpace", "true") \
            .option("ignoreTrailingWhiteSpace", "true") \
            .csv(self.path)

        # Casting label a intero e rimozione righe invalide (label null)
        self.df = self.df.withColumn("label", col("label").cast(IntegerType())) \
                         .filter(col("label").isNotNull()) \
                         .fillna("") # Riempie eventuali buchi nel testo/titolo

        self.df.cache()
        return self.df.count()

    def get_top_users(self, limit=30):
        """
        Aggregazione per identificare gli utenti con maggior volume di caratteri.
        """
        # Somma lunghezza title + text
        return self.df.withColumn("text_length", length(col("text")) + length(col("title"))) \
            .groupBy("id", "label") \
            .agg(sum("text_length").alias("total_length")) \
            .orderBy(desc("total_length")) \
            .limit(limit) \
            .toPandas()
    
    def get_label_length_stats(self):
        """
        Statistiche sulla lunghezza media dei post per classe.
        """
        return self.df.withColumn("text_length", length(col("text")) + length(col("title"))) \
            .groupBy("label") \
            .agg(avg("text_length").alias("normalized_volume")) \
            .orderBy("label") \
            .toPandas()
    
    def correlazione_orario_etichetta(self):
        """
        Calcola la distribuzione oraria normalizzata per classe.
        Normalizzazione necessaria per confrontare classi sbilanciate.
        """
        # Parsing sicuro del timestamp
        df_clean = self.df.filter((col("date").isNotNull()) & (col("date") != "")) \
            .withColumn("ts", try_to_timestamp(col("date"), lit("yyyy-MM-dd HH:mm:ss"))) \
            .filter(col("ts").isNotNull())

        df_hour = df_clean.withColumn("hour", hour(col("ts")))

        # Conteggio grezzo per (Label, Ora)
        hourly_counts = df_hour.groupBy("label", "hour").count()

        # Calcola il totale post per ogni label (indipendentemente dall'ora)
        # per ottenere la percentuale relativa all'interno della stessa classe.
        window_spec = Window.partitionBy("label")

        normalized_df = hourly_counts.withColumn(
            "percentage",
            (col("count") / sum("count").over(window_spec)) * 100
        ).orderBy("hour")
        
        return normalized_df.toPandas()
    
    def get_feature_importance_data(self):
        """
        Wrapper per l'estrazione dei coefficienti dal modello ML (gestito da AI_ML_LLM).
        """
        df_importance = self.ai.extract_feature_importance()
        return df_importance.sort_values(by="coefficient", ascending=False)
    
    def search_keywords(self, keyword, limit):
        """
        Ricerca case-insensitive nel titolo e nel corpo del testo.
        """
        kw = keyword.lower()
        
        results = self.df.filter(
            (lower(col("title")).contains(kw)) |
            (lower(col("text")).contains(kw))
        )

        return results.select("id", "title", "date", "text", "label").limit(limit).toPandas()
    
    def count_labels(self):
        return self.df.groupBy("label").count().orderBy("label").toPandas()
    
    def get_eda_features(self, sample_size=5000):
        """
        Estrae un campione casuale di feature (Lunghezza, Ora) per i boxplot/violin plot.
        """
        df_features = self.df.withColumn("total_len", length(col("title")) + length(col("text"))) \
                            .withColumn("ts", try_to_timestamp(col("date"), lit("yyyy-MM-dd HH:mm:ss"))) \
                            .withColumn("hour", hour(col("ts")))
        
        # Filtro nulli e campionamento random
        return df_features.filter(col("ts").isNotNull()) \
                        .orderBy(rand()) \
                        .select("label", "total_len", "hour") \
                        .limit(sample_size) \
                        .toPandas()
    
    def get_label_variance(self):
        """
        Calcola la varianza della lunghezza dei testi (indice di eterogeneità dello stile).
        """
        return self.df.withColumn("text_len", length(col("text")) + length(col("title"))) \
                  .groupBy("label") \
                  .agg(variance("text_len").alias("variance")) \
                  .orderBy("label") \
                  .toPandas()
    
    def perform_topic_modeling(self, target_label=None, num_topics=5, num_words=10):
        return AI.perform_topic_modeling_LDA(self, target_label, num_topics, num_words)

    def get_self_reference_analysis(self):
        """
        Analisi psicolinguistica: Calcolo ratio pronomi 1^ persona singolare (I-usage).
        """
        # Regex boundaries (\b) per evitare match parziali (es. 'mine' in 'mineral')
        pronoun_regex = r"\b(i|me|my|myself|mine)\b"
        
        # Calcolo occorrenze pronomi vs totale parole (approssimato split spazi)
        df_pronomi = self.df.withColumn("pronoun_count", size(split(lower(col("text")), pronoun_regex)) - 1) \
                             .withColumn("word_count", size(split(col("text"), " ")))
        
        # Filtro per evitare divisioni per zero e calcolo ratio
        df_pronomi = df_pronomi.filter(col("word_count") > 0) \
                               .withColumn("self_ref_ratio", col("pronoun_count") / col("word_count"))
        
        return df_pronomi.groupBy("label").agg(avg("self_ref_ratio").alias("avg_self_ref")).toPandas()
    

    def get_posts_over_time(self):
        """
        Restituisce il conteggio giornaliero dei post (Volume Totale).
        La base dati per il filtraggio temporale della dashboard.
        """
        # Conversione stringa -> data e pulizia null
        df_date = self.df.withColumn("ts", try_to_timestamp(col("date"), lit("yyyy-MM-dd HH:mm:ss"))) \
                        .filter(col("ts").isNotNull()) \
                        .withColumn("post_date", to_date(col("ts")))
        
        # Aggregazione per singola data (Volume Totale aggregato)
        return df_date.groupBy("post_date") \
                    .agg(count("*").alias("post_count")) \
                    .orderBy("post_date") \
                    .toPandas()

    def start_streaming_analytics(self):

        stream_path = "stream_input"
    
        if not os.path.exists(stream_path):
            os.makedirs(stream_path)
            #crea un file dummy vuoto per "inizializzare" lo schema
            with open(os.path.join(stream_path, ".init"), "w") as f:
                f.write("")

        # Definiamo lo schema esatto 
        schema = self.df.schema 

        # Configurazione sorgente streaming
        raw_stream = self.spark.readStream \
            .schema(schema) \
            .option("maxFilesPerTrigger", 1) \
            .csv("stream_input")

        # Conteggio real-time per label
        streaming_counts = raw_stream.groupBy("label").count()

        # permette a Streamlit di fare query SQL
        # outputMode "complete" ricalcola l'intera tabella ogni volta che arrivano i dati
        # format "memory" mantiene i dati in memoria
        # query name "real_time_counts" nome tabella 
        query = streaming_counts.writeStream \
            .outputMode("complete") \
            .format("memory") \
            .queryName("real_time_counts") \
            .start()
        
        return query
    
    def generate_response(self, prompt):
        """
        Interfaccia con il modulo AI per generare risposte LLM.
        """
        return self.ai.generate_response(prompt)
    
    def name_topics_with_llm(self, topics):
        """
        Utilizza l'LLM per assegnare nomi significativi ai topic estratti.
        """
        return self.ai.name_topics(topics)
    