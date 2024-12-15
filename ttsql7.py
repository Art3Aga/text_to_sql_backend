from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Paso 1: Cargar el modelo y el tokenizador preentrenados
model_name = "rakeshkiriyath/gpt2Medium_text_to_sql"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Asegurarse de que el token de padding esté definido
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Paso 2: Crear el conjunto de datos de entrenamiento
training_data = [
    {"input": "Give me the names of all products", "output": "SELECT nombre FROM Productos;"},
    {"input": "What is the price of the product with id 5?", "output": "SELECT precio FROM Productos WHERE id_producto = 5;"},
    {"input": "Show all products with a price greater than 100", "output": "SELECT * FROM Productos WHERE precio > 100;"},
    {"input": "c", "output": "SELECT * FROM Productos WHERE precio < 50;"},
    {"input": "How many products have a price greater than 200?", "output": "SELECT COUNT(*) FROM Productos WHERE precio > 200;"}
]

# Función para tokenizar y alinear entradas y etiquetas
def preprocess_data(data):
    tokenized_data = []
    for item in data:
        input_text = f"Translate to SQL: {item['input']}"
        output_text = item["output"]
        # Concatenar input y output para el aprendizaje autoregresivo
        full_text = input_text + tokenizer.eos_token + output_text
        tokenized = tokenizer(full_text, truncation=True, max_length=128, padding="max_length")
        # Las etiquetas son iguales a los input_ids pero ignoramos los valores de padding
        labels = tokenized["input_ids"].copy()
        labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
        tokenized_data.append({"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"], "labels": labels})
    return tokenized_data

# Preprocesar los datos
tokenized_training_data = preprocess_data(training_data)

# Crear el dataset Hugging Face
dataset = Dataset.from_list(tokenized_training_data)

# Paso 3: Configurar los parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="./finetuned_sql_model",  # Carpeta para guardar el modelo ajustado
    per_device_train_batch_size=4,      # Tamaño del lote
    num_train_epochs=3,                 # Número de épocas
    save_steps=100,                     # Guardar cada 100 pasos
    save_total_limit=2,                 # Mantener solo los 2 últimos modelos guardados
    logging_dir="./logs",               # Carpeta de logs
    evaluation_strategy="no",           # Sin evaluación durante el entrenamiento
    learning_rate=5e-5,                 # Tasa de aprendizaje
    warmup_steps=100,                   # Número de pasos de calentamiento
    weight_decay=0.01,                  # Decaimiento del peso
    gradient_accumulation_steps=1       # Acumulación de gradientes
)

# Crear un Data Collator para manejar el padding dinámico durante el entrenamiento
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # No estamos usando modelado de lenguaje enmascarado
)

# Paso 4: Entrenar el modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()

# Paso 5: Guardar el modelo ajustado
model.save_pretrained("./finetuned_sql_model")
tokenizer.save_pretrained("./finetuned_sql_model")

# Paso 6: Función para generar consultas SQL
def generate_sql(question, model, tokenizer):
    prompt = f"Translate to SQL: {question}"
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    attention_mask = input_ids.ne(tokenizer.pad_token_id)  # Generar atención explícita
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=256,
        num_beams=4,
        early_stopping=True
    )
    sql_query = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Eliminar el prompt de la salida generada
    sql_query = sql_query.replace(prompt, "").strip()
    return sql_query

# Ejemplo de uso
question = "How many products have a price greater than 200?"
sql_query = generate_sql(question, model, tokenizer)
print("Generated SQL:", sql_query)
