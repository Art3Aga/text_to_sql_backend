from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch
import json, random

# Cargar el modelo y el tokenizador preentrenado
model_name = "cssupport/t5-small-awesome-text-to-sql"
config = T5Config.from_pretrained("cssupport/t5-small-awesome-text-to-sql", dropout_rate=0.1)
model = T5ForConditionalGeneration.from_pretrained("cssupport/t5-small-awesome-text-to-sql", config=config)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)  # Activar comportamiento nuevo


# Esquema de la base de datos
schema_context = """
Schema:
Table: productos
Columns:
  - id_producto (integer, primary key)
  - nombre (varchar(100))
  - precio (double)
  - fecha_ingreso (datetime)
"""
training_data = []
training_data_spanish = []

# Función para preprocesar datos
def preprocess_data(data, schema_context):
    tokenized_data = []
    for entry in data:
        input_text = f"{schema_context}\nQuestion: {entry['input']}"
        output_text = entry["output"]
        # Tokenizar entrada y salida
        inputs = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        outputs = tokenizer(
            output_text,
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        # Ajustar las etiquetas (ignorar el padding)
        labels = outputs["input_ids"]
        labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
        tokenized_data.append({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        })
    return tokenized_data

# Datos de entrenamiento y evaluación
with open("dataset.json", "r", encoding="utf-8") as file:
    training_data = json.load(file)
    
with open("dataset_spanish.json", "r", encoding="utf-8") as file:
    training_data_spanish = json.load(file)
    
    
combined_training_data = training_data + training_data_spanish

random.shuffle(combined_training_data)

eval_data = [
    {"input": "How many products have a price less than 50?", 
     "output": "SELECT COUNT(*) FROM productos WHERE precio < 50;"},
    {"input": "¿Cuántos productos tienen un precio menor a 50?",
    "output": "SELECT COUNT(*) FROM productos WHERE precio < 50;"},
    {"input": "Show me the names of the cheapest products", 
     "output": "SELECT nombre FROM productos ORDER BY precio ASC LIMIT 5;"},
    {"input": "Muéstrame los nombres de los productos más baratos",
    "output": "SELECT nombre FROM productos ORDER BY precio ASC LIMIT 5;"},
    {"input": "Which products were added yesterday?", 
     "output": "SELECT * FROM productos WHERE DATE(fecha_ingreso) = CURDATE() - INTERVAL 1 DAY;"},
    {"input": "¿Qué productos se agregaron ayer?",
"output": "SELECT * FROM productos WHERE DATE(fecha_ingreso) = CURDATE() - INTERVAL 1 DAY;"},
    {"input": "What is the name of the product with the highest price?", 
     "output": "SELECT nombre FROM productos ORDER BY precio DESC LIMIT 1;"},
    {"input": "¿Cuál es el nombre del producto con el precio más alto?",
"output": "SELECT nombre FROM productos ORDER BY precio DESC LIMIT 1;"},
    {"input": "List all products and their details added this year", 
     "output": "SELECT * FROM productos WHERE YEAR(fecha_ingreso) = YEAR(CURDATE());"},
    {"input": "Enumera todos los productos y sus detalles agregados este año",
"output": "SELECT * FROM productos WHERE YEAR(fecha_ingreso) = YEAR(CURDATE());"},
    {"input": "What is the total number of products in the database?", 
     "output": "SELECT COUNT(*) FROM productos;"},
    {"input": "¿Cuál es el número total de productos en la base de datos?",
"output": "SELECT COUNT(*) FROM productos;"},
    {"input": "Show the names and prices of products priced between 50 and 150", 
     "output": "SELECT nombre, precio FROM productos WHERE precio BETWEEN 50 AND 150;"},
    {"input": "Mostrar los nombres y precios de los productos con precio entre 50 y 150",
"output": "SELECT nombre, precio FROM productos WHERE precio BETWEEN 50 AND 150;"},
    {"input": "List all products sorted by date added in descending order", 
     "output": "SELECT * FROM productos ORDER BY fecha_ingreso DESC;"},
    {"input": "Listar todos los productos ordenados por fecha de adición en orden descendente",
"output": "SELECT * FROM productos ORDER BY fecha_ingreso DESC;"},
    {"input": "How many products have been added this week?", 
     "output": "SELECT COUNT(*) FROM productos WHERE WEEK(fecha_ingreso) = WEEK(CURDATE()) AND YEAR(fecha_ingreso) = YEAR(CURDATE());"},
    {"input": "¿Cuántos productos se han añadido esta semana?",
"output": "SELECT COUNT(*) FROM productos WHERE WEEK(fecha_ingreso) = WEEK(CURDATE()) AND YEAR(fecha_ingreso) = YEAR(CURDATE());"},
    {"input": "What is the price of the product with ID 15?", 
     "output": "SELECT precio FROM productos WHERE id_producto = 15;"},
    {"input": "¿Cuál es el precio del producto con ID 15?",
"output": "SELECT precio FROM productos WHERE id_producto = 15;"},
]


# training_data.extend(additional_training_data)

# Preprocesar datos
train_dataset = Dataset.from_list(preprocess_data(combined_training_data, schema_context))
eval_dataset = Dataset.from_list(preprocess_data(eval_data, schema_context))

# Configurar los argumentos del entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_t5_spanish",
    per_device_train_batch_size=4,  # Tamaño del lote reducido si tienes poca memoria GPU
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    eval_steps=10,  # Usar eval_steps en lugar de evaluation_strategy (descontinuado)
    save_strategy="epoch",        # Guardar al final de cada época
    predict_with_generate=True,   # Activar generación en evaluación
    generation_max_length=128,    # Longitud máxima para la generación
    generation_num_beams=8,       # Número de beams para búsqueda
    learning_rate=2e-5,
    warmup_steps=135,
    weight_decay=0.01,
    save_total_limit=2
)

# Usar un DataCollator para manejar el padding dinámico
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Función para calcular métricas
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Eliminar padding y limpiar textos
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # Calcular exactitud
    exact_matches = sum([pred == label for pred, label in zip(decoded_preds, decoded_labels)])
    accuracy = exact_matches / len(decoded_preds) if len(decoded_preds) > 0 else 0

    return {"accuracy": accuracy}

# Configurar el Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Entrenar el modelo
# trainer.train()

# # Guardar el modelo ajustado
# model.save_pretrained("./fine_tuned_t5_spanish")
# tokenizer.save_pretrained("./fine_tuned_t5_spanish")

tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_t5_spanish")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_t5_spanish")
model = model.to(device)
model.eval()

# Función para generar consultas SQL
def generate_sql(question):
    input_text = f"{schema_context}\nQuestion: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        model.to("cuda")
    output = model.generate(input_ids, max_length=128, num_beams=7, early_stopping=True)
    sql_query = tokenizer.decode(output[0], skip_special_tokens=True)
    return sql_query

# Prueba con una pregunta
question = "How many products have a price greater than 200?"
sql_query = generate_sql(question)
print("Generated SQL:", sql_query)
