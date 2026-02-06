import pandas as pd
import time
import os

SOURCE_CSV = "all_user_with_anorexia.csv"
STREAM_DIR = "stream_input"

if not os.path.exists(STREAM_DIR):
    os.makedirs(STREAM_DIR)

print("Avvio simulazione stream. Premi Ctrl+C per fermare.")

# 1. Caricamento e Randomizzazione Totale
df = pd.read_csv(SOURCE_CSV)
df_shuffled = df.sample(frac=1).reset_index(drop=True)

batch_size = 20 

try:
    for i in range(0, len(df_shuffled), batch_size):
        chunk = df_shuffled.iloc[i:i+batch_size]
        
        # Uso un timestamp nel nome file per evitare collisioni
        timestamp = int(time.time() * 1000)
        file_path = f"{STREAM_DIR}/batch_{timestamp}.csv"
        
        # Scrittura del chunk
        chunk.to_csv(file_path, index=False)
        print(f"Inviato batch {i//batch_size + 1} ({batch_size} righe randomizzate)...")
        
        time.sleep(3) 
except KeyboardInterrupt:
    print("\nSimulazione terminata.")