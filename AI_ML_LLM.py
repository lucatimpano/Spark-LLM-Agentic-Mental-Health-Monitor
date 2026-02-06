# AI_ML_LLM.py
import os
import pandas as pd
import streamlit as st
from google import genai
from pyspark.ml import PipelineModel
from pyspark.ml import Pipeline
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
# Importazioni PySpark Core & SQL
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, lower, regexp_replace, trim
)
import time
from google.api_core import exceptions

class AI:
    def __init__(self):
        self.zero_shot_pipeline = None

    @staticmethod
    def generate_response(prompt, max_retries=3):
        """
        Genera una risposta tramite Google Gemini (LLM) fornendo il contesto
        di esecuzione PySpark e Streamlit.
        """
        # Recupero della chiave API
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        
        # Prompt di Sistema:
        context = """
            Sei un esperto Senior in PySpark e Streamlit. 
            Il tuo obiettivo è generare codice ROBUSTO e pronto all'esecuzione.
            
            VARIABILI DISPONIBILI:
            - 'df': Spark DataFrame con colonne ['id', 'title', 'date', 'text', 'label'].
            - 'spark': SparkSession attiva.
            - 'st': Libreria Streamlit.
            - 'px': Plotly Express.
            - 'proc': Istanza di LogicaReddit con metodi predefiniti.

            NOTE:
            Le label sono 3:
            - 0: Control (Sani)
            - 1: Depression
            - 2: Anorexia
            Il dataset è sbilanciato verso la classe 0 (Control), e la lingua utilizzata è l'inglese.

            REGOLE DI CODIFICA (CRITICHE):
            1. GESTIONE DATI SPORCHI: Il dataset contiene stringhe vuote ('') e malformate. 
               NON usare funzioni temporali dirette su 'date'.
               USA SEMPRE: `try_to_timestamp(col("date"), lit("yyyy-MM-dd HH:mm:ss"))`.
            2. FILTRO NULL: Dopo ogni cast, filtra SEMPRE i valori nulli: 
               `.filter(col("nuova_colonna").isNotNull())`.
            3. OUTPUT: Genera il codice esclusivamente tra tag ```python ... ```.
            4. VISUALIZZAZIONE: 
               - Usa `st.plotly_chart(fig)` per i grafici.
               - Verifica sempre `if not pdf.empty:` prima di plottare.
            5. PERFORMANCE: Non chiamare mai .toPandas() su dataset interi. Aggrega prima.
            """
    
        full_prompt = f"{context}\n\nUser: {prompt}"
        
        try:
            # Chiamata al modello
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=full_prompt,
            )
            return response.text
        
        # Gestione errori con retry
        except exceptions.ServiceUnavailable:
            wait_time = (2 ** i)  # Backoff esponenziale: 1s, 2s, 4s...
            time.sleep(wait_time)
            if i == max_retries - 1:
                return "Errore: Il server AI è attualmente sovraccarico. Riprova tra poco."
                
        except Exception as e:
            return f"Errore imprevisto: {str(e)}"
    
    def extract_feature_importance(self):
        """
        Carica un modello Pipeline PySpark pre-addestrato ed estrae
        i coefficienti delle parole per ogni classe.
        """
        model_path = "models/lr_pipeline_model"

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato in {model_path}. Esegui prima l'addestramento.")
        
        # Caricamento della Pipeline completa salvata su disco
        model = PipelineModel.load(model_path)
        
        # elenco parole mappate su indici
        vocab = model.stages[3].vocabulary
        
        # contiene le etichette delle classi originali
        labels = model.stages[0].labels
        
        # contiene la matrice dei coefficienti
        lr_model = model.stages[-1]
        
        # La matrice dei coefficienti ha dimensione (Num_Classi, Num_Parole)
        # Convertiamo la matrice in array numpy per iterare
        coefficients = lr_model.coefficientMatrix.toArray()
        
        # Per la visualizzazione
        importance_data = []
        
        # Iteriamo su ogni classe
        for i, class_label in enumerate(labels):
            # Associamo ogni parola del vocabolario al suo peso specifico per quella classe
            for word, weight in zip(vocab, coefficients[i]):
                importance_data.append({
                    "word": word, 
                    "coefficient": weight, 
                    "class": class_label
                })
            
        return pd.DataFrame(importance_data)
    
    def perform_topic_modeling_LDA(self, target_label=None, num_topics=5, num_words=10):
        """
        Pipeline NLP completa per Latent Dirichlet Allocation (LDA).
        Fasi: Cleaning -> Tokenization -> StopWords Removal -> CountVectorizer -> LDA.
        """
        # 1. Converto per sicurezza la label in intero e filtro
        if target_label is not None:
            df_filtered = self.df.filter(col("label") == int(target_label))
        else:
            df_filtered = self.df

        if df_filtered.count() == 0:
            return pd.DataFrame()

        # 2. Pre-processing testuale
        # Rimozione URL, tag rimossi/cancellati e caratteri non alfabetici
        df_clean = df_filtered.withColumn("clean_text", lower(col("text")))
        df_clean = df_clean.withColumn("clean_text", regexp_replace("clean_text", r"http\S+", "")) \
                           .withColumn("clean_text", regexp_replace("clean_text", r"\[removed\]|\[deleted\]", "")) \
                           .withColumn("clean_text", regexp_replace("clean_text", r"[^a-z\s]", " ")) \
                           .withColumn("clean_text", trim("clean_text"))

        # 3. Costruzione Pipeline ML
        tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="words", pattern="\\s+", minTokenLength=4)
        remover = StopWordsRemover(inputCol="words", outputCol="filtered")
        
        # Estensione stop words standard
        base_stops = remover.getStopWords()
        custom_stops = base_stops + ["like", "just", "know", "think", "really", "want", "people", "make", "good", "time"]
        remover.setStopWords(custom_stops)

        # CountVectorizer: Limitiamo il vocabolario per performance e rumore
        # minDF=2.0 ignora termini che appaiono in meno di 2 documenti
        # maxDF=0.90 ignora termini che appaiono in più del 90% dei documenti
        cv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=1000, minDF=2.0, maxDF=0.90)

        lda = LDA(k=num_topics, maxIter=20)
        pipeline = Pipeline(stages=[tokenizer, remover, cv, lda])
        
        # 4. Training e Estrazione
        model = pipeline.fit(df_clean)
        
        # Estrazione modello LDA
        lda_model = model.stages[-1]    

        # Vocabolario da CountVectorizer
        cv_model = model.stages[-2]

        # 5. Estrazione termini principali per topic
        vocab = cv_model.vocabulary
        
        # Estrazione termini principali per topic
        topics = lda_model.describeTopics(num_words)
        topic_data = []
        topics_rdd = topics.collect()
        
        for row in topics_rdd:
            topic_id = row['topic']     # ID del topic
            term_indices = row['termIndices']      # Indici dei termini principali
            terms = [vocab[idx] for idx in term_indices]    # Mappatura indici -> termini
            topic_data.append({"topic_id": topic_id, "terms": ", ".join(terms)})    # Aggiunta al risultato
            
        return pd.DataFrame(topic_data)
    
    @staticmethod
    def name_topics(topics_df):
        """
        Utilizzo di un LLM per interpretare semanticamente i gruppi di termini
        generati dall'algoritmo LDA e assegnare un titolo sintetico.
        """
        # Costruzione di una stringa di contesto leggibile per l'LLM
        # Topic 0: term1, term2, term3 \n Topic 1: term1, term2, term3 \n ...
        topics_str = ""
        for index, row in topics_df.iterrows():
            topics_str += f"Topic {row['topic_id']}: {row['terms']}\n"
        
        prompt = f"""
        Sei un analista semantico. Ecco dei topic estratti tramite LDA da post di Reddit (contesto salute mentale/generico).
        
        {topics_str}
        
        Il tuo compito:
        Per ogni Topic, genera un "Titolo Breve" (max 3-4 parole) che riassuma il tema.
        Restituisci SOLO una lista formattata così:
        0: Titolo
        1: Titolo
        ...
        """
        
        try:
            # Inizializzazione client per richiesta specifica
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            return f"Errore generazione nomi: {e}"
        
    