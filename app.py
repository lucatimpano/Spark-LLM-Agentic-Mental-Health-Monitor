# import
import os
import re
import psutil
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

# Importazioni PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, try_to_timestamp, to_date, count, length, sum, desc, avg, rand

# Moduli locali
from AI_ML_LLM import AI
from logica import LogicaReddit


# 1. CONFIGURAZIONE E COSTANTI


# Configurazione base della pagina Streamlit
st.set_page_config(
    page_title="Progetto Big Data",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Percorsi del file system
BASE = "/home/luca/Scrivania/UNI/Magistrale/BigData/progetto"
CSV_PATH = os.path.join(BASE, "all_user_with_anorexia.csv")

# Mappatura delle etichette
LABEL_MAP = {
    0: "Control (Sani)",
    1: "Depression",
    2: "Anorexia",
    "0": "Control (Sani)", 
    "1": "Depression",
    "2": "Anorexia"
}

# colori per i grafici
COLOR_MAP = {
    "Control (Sani)": "#2ecc71",
    "Depression": "#3498db",      
    "Anorexia": "#e74c3c"         
}


# 2. INIZIALIZZAZIONE

@st.cache_resource
def init_processor():
    """
    Inizializza la classe di logica e carica i dati in memoria.
    """
    # Per il sistema distribuito
    path = "/tmp/all_user_with_anorexia.csv"
    if not os.path.exists(path):
        path = "all_user_with_anorexia.csv"
    
    proc_instance = LogicaReddit(path)
    proc_instance.load_data()
    return proc_instance

# Istanziazione del processore globale
proc = init_processor()

# Variabili di contesto per l'esecuzione dinamica del codice 
context_vars = {"spark": proc.spark, "df": proc.df, "st": st, "px": px, "pd": pd}


# 3. FUNZIONI DI UTILITÀ E VISUALIZZAZIONE


def apply_labels(df, col_name='label'):
    """
    Converte le etichette numeriche in nomi
    """
    df['label_name'] = df[col_name].map(LABEL_MAP)
    return df

@st.fragment(run_every=2)
def render_system_monitor():
    """
    Monitora le risorse di sistema in tempo reale.
    Il decoratore @st.fragment permette l'aggiornamento del widget 
    """
    st.subheader("Risorse di Sistema")
    
    cpu_p = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_p = ram.percent
    
    col_cpu, col_ram = st.columns(2)
    col_cpu.metric("CPU", f"{cpu_p}%")
    col_ram.metric("RAM", f"{ram_p}%")

def execute_spark_code(response_text, spark_env):
    """
    Esegue dinamicamente codice Python/Spark generato dall'LLM.
        
    Reitorna una tupla: (bool success, str result/error)
    """
    # Regex per estrarre il blocco di codice delimitato da markdown ```python ... ```
    code_match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
    
    if code_match:
        code = code_match.group(1)
        try:
            # exec() esegue il codice nel contesto fornito
            exec(code, {}, spark_env)
            return True, code
        except Exception as e:
            return False, str(e)
    return False, None


# 4. INTERFACCIA UTENTE: SIDEBAR E NAVIGAZIONE


with st.sidebar:
    st.image("logo_project.png", width=250)
    
    st.markdown("---")
    render_system_monitor()
    
    st.title("Navigazione")
    st.markdown("---")
    
    # Menu di navigazione principale
    page = option_menu(
        menu_title=None,
        options=["Home", "Temporal Analysis", "Content Search", "AI & ML", "Streaming Real-Time"],
        icons=["house", "clock-history", "search", "robot", "cloud-upload"],
        menu_icon="cast", 
        default_index=0,
    )


# 5. PAGINA: HOME (Overview e Analisi Statistica)


if page == "Home":
    st.title("Overview Dataset Reddit Posts")
    st.markdown("Dashboard di monitoraggio delle metriche principali e della distribuzione delle classi.")
    
    # METRICHE PRINCIPALI 
    col_metrica1, col_metrica2 = st.columns(2)
    col_metrica1.metric("Totale Post nel Dataset", proc.df.count())
    col_metrica2.metric("Utenti Unici Censiti", proc.df.select("id").distinct().count())

    st.markdown("---")
    
    # BILANCIAMENTO DELLE CLASSI
    st.header("1. Analisi del Bilanciamento delle Classi")
    st.write("""
    In questa sezione confrontiamo la distribuzione numerica dei post (frequenza) con il volume effettivo di testo generato.
    """)
    
    col_bal1, col_bal2 = st.columns(2)
    
    with col_bal1:
        st.subheader("Frequenza Numerica")
        counts_df = proc.count_labels()
        counts_df = apply_labels(counts_df)
        fig_count = px.pie(
            counts_df,
            values="count",
            names="label_name",
            title="Ripartizione percentuale dei Post",
            color="label_name",
            color_discrete_map=COLOR_MAP,
            template="plotly_white"
        )
        fig_count.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_count, width='content')
        
    with col_bal2:
        st.subheader("Volume Testuale")
        stats_df = proc.get_label_length_stats()
        stats_df = apply_labels(stats_df)
        
        fig_pie = px.pie(
            stats_df, 
            values="normalized_volume", 
            names="label_name",
            title="Peso Volumetrico (Lunghezza Media)",
            color="label_name",
            color_discrete_map=COLOR_MAP,
            template="plotly_white"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, width='content')

    st.markdown("---")

    # DISTRIBUZIONE STATISTICA (BoxPlot)
    st.header("2. Analisi della Distribuzione delle Lunghezze")
    st.write("""
    Il seguente Box-Plot su scala logaritmica evidenzia la dispersione della lunghezza dei post per ogni categoria.
    """)
    
    df_eda = proc.get_eda_features()
    df_eda = apply_labels(df_eda)
    fig_len = px.box(
        df_eda, 
        x="label_name", 
        y="total_len", 
        color="label_name",
        title="Distribuzione Lunghezza Post (Scala Log)",
        labels={"total_len": "Caratteri (Titolo + Testo)", "label_name": "Categoria"},
        log_y=True 
    )
    st.plotly_chart(fig_len, use_container_width=True)

    # Sezione Varianza (Tabellare)
    c_desc, c_table = st.columns([1, 1])
    with c_desc:
        st.markdown("### Indicatori di Varianza")
        st.write("""
        La tabella a fianco mostra la varianza della lunghezza dei testi per classe. 
        Una varianza elevata indica una minore omogeneità nello stile di scrittura degli utenti appartenenti a quella categoria.
        """)
    with c_table:
        var_df = proc.get_label_variance()
        var_df = apply_labels(var_df)
        st.table(var_df[['label_name', 'variance']].rename(columns={
            'label_name': 'Categoria',
            'variance': 'Varianza (σ²)'
        }))

    st.markdown("---")
    
    # ANALISI UTENTI (Top Charts)
    st.header("3. Analisi Comportamentale Utenti")
    st.write("Identificazione degli utenti più attivi in termini di volume di caratteri prodotti (Top 30).")
    
    top_users = proc.get_top_users()
    top_users = apply_labels(top_users)
    top_users = top_users.sort_values(by="total_length", ascending=False)

    fig_top = px.bar(
        top_users, 
        x="id", 
        y="total_length",
        title="Ranking Utenti per Volume di Testo",
        labels={"total_length": "Caratteri Totali", "id": "User ID", "label_name": "Categoria"},
        color="label_name",
        color_discrete_map=COLOR_MAP,
        template="plotly_white",
        hover_data=["label_name"]
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")
    
    # ANALISI LINGUISTICA
    st.header("4. Approfondimento Linguistico")
    st.subheader("Indice di Focalizzazione Interna")
    
    st.info("""
    In psicologia, un alto uso di self-reference è correlato a disturbi dell'umore.
    """)

    if st.button("Esegui Analisi Self-Reference"):
        with st.spinner("Elaborazione..."):
            self_ref_df = proc.get_self_reference_analysis()
            self_ref_df = apply_labels(self_ref_df)
            
            fig_self = px.bar(
                self_ref_df,
                x="label_name",
                y="avg_self_ref",
                color="label_name",
                title="Self-Reference Ratio Medio per Categoria",
                labels={
                    "avg_self_ref": "Ratio (Pronomi 1ª Pers / Totale)",
                    "label_name": "Categoria"
                },
                color_discrete_map=COLOR_MAP,
                template="plotly_white"
            )
            
            fig_self.update_layout(yaxis_tickformat='.2%')
            st.plotly_chart(fig_self, use_container_width=True)


# 6. PAGINA: TEMPORAL ANALYSIS


elif page == "Temporal Analysis":
    st.title("Analisi Temporale")
    st.header("Ritmi Circadiani")
    
    st.write("Analisi della distribuzione dei post durante le 24 ore per identificare pattern di attività notturna o diurna.")
    
    df_norm = proc.correlazione_orario_etichetta()
    
    fig_circadian = px.line(
        df_norm, 
        x="hour", 
        y="percentage", 
        color="label",
        markers=True,
        title="Ritmo Circadiano: Depressione vs Controllo",
        labels={
            "hour": "Orario", 
            "percentage": "Frequenza Relativa (%)",
            "label": "Gruppo"
        },
        template="plotly_white"
    )

    # Configurazione asse X per mostrare tutte le ore (0-23)
    fig_circadian.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    st.plotly_chart(fig_circadian, use_container_width=True)

    st.header("Densità di Pubblicazione")
    st.write("Distribuzione della densità di pubblicazione per ora del giorno divisa per categoria.")
    
    df_eda = proc.get_eda_features()
    df_eda = apply_labels(df_eda)
    
    fig_hour = px.violin(
        df_eda, x="label_name", y="hour", 
        color="label_name",
        box=True, points="all",
        title="Densità Oraria per Classe",
        labels={"hour": "Ora del Giorno", "label_name": "Categoria"}
    )
    st.plotly_chart(fig_hour, use_container_width=True)

    #Visualizzazione Serie Storica
    st.header("Trend Temporale")
    
    with st.spinner("Estrazione dati dal cluster..."):
        trend_df = proc.get_posts_over_time()
        trend_df['post_date'] = pd.to_datetime(trend_df['post_date'])

    view_mode = st.radio(
        "Seleziona l'ampiezza della visualizzazione:",
        ["Tutta la Timeline", "Anno Specifico", "Mese Specifico"],
        horizontal=True
    )

    # Variabili per il calcolo dei Delta
    curr_total = 0
    prev_total = 0
    delta_val = None

    if view_mode == "Anno Specifico":
        years = sorted(trend_df['post_date'].dt.year.unique(), reverse=True)
        sel_year = st.selectbox("Seleziona l'anno:", years)
        
        # Filtro anno corrente e precedente
        curr_df = trend_df[trend_df['post_date'].dt.year == sel_year]
        prev_df = trend_df[trend_df['post_date'].dt.year == (sel_year - 1)]
        
        curr_total = curr_df['post_count'].sum()
        prev_total = prev_df['post_count'].sum()
        filtered_df = curr_df

    elif view_mode == "Mese Specifico":
        years = sorted(trend_df['post_date'].dt.year.unique(), reverse=True)
        sel_year = st.selectbox("1. Seleziona l'anno:", years)
        available_months = sorted(trend_df[trend_df['post_date'].dt.year == sel_year]['post_date'].dt.month.unique())
        sel_month = st.selectbox("2. Seleziona il mese:", available_months)
        
        # Filtro mese corrente vs stesso mese anno precedente (YoY)
        curr_df = trend_df[(trend_df['post_date'].dt.year == sel_year) & (trend_df['post_date'].dt.month == sel_month)]
        prev_df = trend_df[(trend_df['post_date'].dt.year == (sel_year - 1)) & (trend_df['post_date'].dt.month == sel_month)]
        
        curr_total = curr_df['post_count'].sum()
        prev_total = prev_df['post_count'].sum()
        filtered_df = curr_df
    
    else: # Tutta la Timeline
        filtered_df = trend_df
        curr_total = trend_df['post_count'].sum()

    # Visualizzazione Metriche
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.metric(label="Volume Post Selezionati", value=f"{curr_total:,}")
    
    with col_m2:
        if view_mode != "Tutta la Timeline":
            if prev_total > 0:
                delta_val = ((curr_total - prev_total) / prev_total) * 100
                st.metric(
                    label="Rispetto all'anno precedente", 
                    value="Percentuale", 
                    delta=f"{delta_val:.2f}%"
                )
            else:
                st.metric(label="Rispetto all'anno precedente", value=f"{curr_total:,}", delta="N/A (No prev data)")

    # Rendering Grafico
    if not filtered_df.empty:
        fig_trend = px.area(
            filtered_df, x="post_date", y="post_count",
            title=f"Analisi Volumetrica: {view_mode}",
            template="plotly_dark", color_discrete_sequence=["#3498db"]
        )
        fig_trend.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_trend, use_container_width=True)


# 7. PAGINA: CONTENT SEARCH


elif page == "Content Search":
    st.title("Esplorazione Contenuti")
    st.header("Ricerca Parole Chiave nei Post")
    
    col_input, col_slider = st.columns([0.7, 0.3])
    
    with col_input:
        search_term = st.text_input("Inserisci parola chiave:")
        
    with col_slider:
        result_limit = st.slider("Limite risultati", 10, 300, 100)
     
    if search_term:
        with st.spinner(f"Ricerca di '{search_term}' nel cluster Spark..."):
            results_df = proc.search_keywords(search_term, result_limit)

            if not results_df.empty:
                st.success(f"Trovati {len(results_df)} post rilevanti")
                st.dataframe(
                    results_df,
                    column_config={
                        "text": st.column_config.TextColumn("Contenuto Post", width="large"),
                        "title": st.column_config.TextColumn("Titolo", width="medium"),
                        "date": "Data",
                        "label": "Etichetta (0,1,2)"
                    },
                    hide_index=True
                )
            else:
                st.warning("Nessun post trovato con questa parola chiave.")


# 8. PAGINA: AI & ML


elif page == "AI & ML":
    st.title("Intelligenza Artificiale e Machine Learning")
    st.markdown("""
        Analisi interazione con il dataset tramite Large Language Models e modelli di AI.
    """)
     
    col_dati, col_chat = st.columns([0.65, 0.35], gap="large")
    
    with st.sidebar:
        st.divider()
        st.subheader("Configurazione Parametri")
        num_words = st.slider("Numero di feature visualizzate", 5, 100, 20)

    # Gestione dello stato per il dataset di importanza
    if "importance_df" not in st.session_state:
        with st.spinner("Calcolo feature importance in corso..."):
            st.session_state.importance_df = proc.get_feature_importance_data()
    
    importance_df = st.session_state.importance_df

    # Analisi Feature Importance
    with col_dati:
        with st.container(border=True):
            st.subheader("Analisi delle Feature")
            st.info("Distribuzione dei coefficienti del modello per la determinazione della rilevanza semantica.")
            
            available_classes = importance_df['class'].unique()
            target_class = st.selectbox(
                "Classe di riferimento:", 
                available_classes
            )

            df_viz = importance_df[importance_df['class'] == target_class]
            df_viz = df_viz.sort_values(by="coefficient", ascending=False).head(num_words)

            tab_tree, tab_bar = st.tabs(["Mappa Gerarchica", "Analisi Lineare"])
        
            with tab_tree:
                fig_tree = px.treemap(
                    df_viz, 
                    path=['word'], 
                    values='coefficient',
                    color='coefficient',
                    color_continuous_scale='Greys',
                    template="plotly_white"
                )
                fig_tree.update_layout(margin=dict(t=5, l=5, r=5, b=5))
                st.plotly_chart(fig_tree, use_container_width=True)

            with tab_bar:
                fig_bar = px.bar(
                    df_viz, 
                    x="coefficient", 
                    y="word", 
                    orientation='h',
                    color='coefficient',
                    color_continuous_scale='Greys'
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    margin=dict(t=5, l=5, r=5, b=5),
                    xaxis_title="Coefficiente",
                    yaxis_title="Token"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    # Assistente AI
    with col_chat:
        with st.container(border=True):
            @st.fragment
            def render_chat_interface():
                st.subheader("Assistente Tecnico")
                
                chat_container = st.container(height=450)

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                with chat_container:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                if prompt := st.chat_input("Inserisci query di analisi..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        response = proc.generate_response(prompt)

                        if "Errore:" in response:
                            st.warning(response)
                        else:
                            with st.chat_message("assistant"):
                                st.markdown(response)
                                success, result = execute_spark_code(response, context_vars)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})

                if st.button("Reset Sessione Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

            render_chat_interface()

    st.divider()
    
    # Analisi dei Topic (LDA)
    with st.container(border=True):
        st.header("Topic Modeling (Latent Dirichlet Allocation)")
        st.write("Estrazione non supervisionata di cluster tematici dal corpus testuale.")
        
        col_set, col_res = st.columns([0.3, 0.7], gap="medium")
        
        with col_set:
            st.markdown("##### Configurazione Algoritmo")
            selected_label_name = st.selectbox(
                "Sottogruppo Analitico:",
                ["Tutti", "Control (Sani)", "Depression", "Anorexia"]
            )
            
            map_inv = {"Control (Sani)": 0, "Depression": 1, "Anorexia": 2, "Tutti": None}
            target_label = map_inv[selected_label_name]
            
            num_topics = st.slider("Numero di Topic (K-Clusters)", 3, 10, 5)
            
            if st.button("Esegui Modellazione", type="primary", use_container_width=True):
                st.session_state.run_lda = True
            else:
                if 'run_lda' not in st.session_state:
                    st.session_state.run_lda = False

        with col_res:
            if st.session_state.run_lda:
                with st.spinner("Elaborazione cluster LDA..."):
                    topics_df = proc.perform_topic_modeling(target_label=target_label, num_topics=num_topics)
                    
                    if not topics_df.empty:
                        st.markdown("**Analisi Semantica Automatica:**")
                        ai_names = proc.name_topics_with_llm(topics_df)
                        st.success(ai_names)
                        
                        with st.expander("Dettagli Matrice Termini"):
                            st.dataframe(topics_df, use_container_width=True)
                    else:
                        st.warning("Volume di dati insufficiente per la convergenza del modello.")


# 9. PAGINA: STREAMING


elif page == "Streaming Real-Time":
    st.header("Live Data Stream (Spark Structured Streaming)")
    
    # Inizializza lo stream se non attivo
    if "stream_query" not in st.session_state:
        st.session_state.stream_query = proc.start_streaming_analytics()

    @st.fragment(run_every=3) # Aggiorna il grafico ogni 3 secondi
    def live_chart():
        # Interroga la tabella in memoria creata dal Sink di Spark
        try:
            pdf = proc.spark.sql("SELECT * FROM real_time_counts").toPandas()
            if not pdf.empty:
                pdf = apply_labels(pdf)
                fig = px.bar(pdf, x="label_name", y="count", color="label_name",
                             title="Distribuzione Classi in Tempo Reale")
                st.plotly_chart(fig, use_container_width=True)
                st.write(f"Ultimo aggiornamento: {pd.Timestamp.now()}")
            else:
                st.info("In attesa di dati dallo stream...")
        except Exception:
            st.error("Tabella streaming non ancora disponibile.")

    live_chart()