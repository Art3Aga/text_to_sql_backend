from transformers import T5ForConditionalGeneration, RobertaTokenizer, TrainingArguments, Trainer
from datasets import Dataset

# Cargar el modelo y el tokenizador de CodeT5
model_name = "Salesforce/codet5-base"  # Modelo preentrenado
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Paso 1: Datos de entrenamiento
training_data = [
    {"input": "Show me all products with a price above 20", "output": "SELECT * FROM productos WHERE precio > 20;"},
    {"input": "List all products with a price below 10", "output": "SELECT * FROM productos WHERE precio < 10;"},
    {"input": "What is the total number of products?", "output": "SELECT COUNT(*) FROM productos;"},
    {"input": "Show the names of products with price 50", "output": "SELECT nombre FROM productos WHERE precio = 50;"},
]

# Paso 2: Convertir los datos a Dataset
dataset = Dataset.from_list([
    {
        "input_ids": tokenizer.encode(entry["input"], truncation=True, padding="max_length", max_length=128),
        "labels": tokenizer.encode(entry["output"], truncation=True, padding="max_length", max_length=128),
    }
    for entry in training_data
])

# Paso 3: Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./fine_tuned_codet5",  # Carpeta para guardar el modelo
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="no",
    save_strategy="epoch"
)

# Paso 4: Configurar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Paso 5: Entrenar el modelo
trainer.train()

# Guardar el modelo ajustado
model.save_pretrained("./fine_tuned_codet5")
tokenizer.save_pretrained("./fine_tuned_codet5")

# Paso 6: Usar el modelo ajustado
# Cargar el modelo ajustado para generar consultas SQL
model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_codet5")
tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_codet5")

def generate_sql(question):
    input_ids = tokenizer.encode(question, return_tensors="pt", max_length=128, truncation=True)
    output = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Ejemplo de uso
question = "Show me all products with a price above 20"
sql_query = generate_sql(question)
print("Generated SQL:", sql_query)
