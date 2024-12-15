# from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
# from datasets import Dataset
# import torch

# # Cargar el modelo y el tokenizador preentrenado
# model_name = "cssupport/t5-small-awesome-text-to-sql"
# model = T5ForConditionalGeneration.from_pretrained(model_name)
# tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)  # Activar comportamiento nuevo

# # Función para preprocesar datos
# def preprocess_data(data):
#     tokenized_data = []
#     for entry in data:
#         input_text = entry["input"]
#         output_text = entry["output"]
#         # Tokenizar entrada y salida
#         inputs = tokenizer(
#             input_text,
#             truncation=True,
#             padding="max_length",
#             max_length=128,
#         )
#         outputs = tokenizer(
#             output_text,
#             truncation=True,
#             padding="max_length",
#             max_length=128,
#         )
#         # Ajustar las etiquetas (ignorar el padding)
#         labels = outputs["input_ids"]
#         labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
#         tokenized_data.append({
#             "input_ids": inputs["input_ids"],
#             "attention_mask": inputs["attention_mask"],
#             "labels": labels,
#         })
#     return tokenized_data

# # Datos de entrenamiento y evaluación
# training_data = [
#     {"input": "Show me the names of products with a price greater than 20", 
#      "output": "SELECT nombre FROM productos WHERE precio > 20;"},
#     {"input": "List all products in stock", 
#      "output": "SELECT * FROM productos WHERE stock > 0;"},
#     {"input": "How many products are there in total?", 
#      "output": "SELECT COUNT(*) FROM productos;"},
#     {"input": "Get the details of products added today", 
#      "output": "SELECT * FROM productos WHERE DATE(fecha_ingreso) = CURDATE();"},
#     {"input": "What is the price of the product called 'Laptop'?", 
#      "output": "SELECT precio FROM productos WHERE nombre = 'Laptop';"},
#     {"input": "List the products with stock less than 10", 
#      "output": "SELECT * FROM productos WHERE stock < 10;"},
#     {"input": "Show me all products ordered by price descending", 
#      "output": "SELECT * FROM productos ORDER BY precio DESC;"},
#     {"input": "What are the details of the product with ID 5?", 
#      "output": "SELECT * FROM productos WHERE id = 5;"},
#      {"input": "how many products are in total?", 
#      "output": "SELECT COUNT(*) FROM productos;"},
# ]

# eval_data = [
#     {"input": "How many products have a price less than 50?", 
#      "output": "SELECT COUNT(*) FROM productos WHERE precio < 50;"},
#     {"input": "List the products that were added yesterday", 
#      "output": "SELECT * FROM productos WHERE DATE(fecha_ingreso) = CURDATE() - INTERVAL 1 DAY;"},
#      {"input": "how many products are in total?", 
#      "output": "SELECT COUNT(*) FROM productos;"},
# ]

# synthetic_data = [
#     {"input": "Which products cost more than 100?", 
#      "output": "SELECT * FROM productos WHERE precio > 100;"},
#     {"input": "How many products cost less than 50", 
#      "output": "SELECT COUNT(*) FROM productos WHERE precio < 50;"},
# ]

# training_data.extend(synthetic_data);

# # Preprocesar datos
# train_dataset = Dataset.from_list(preprocess_data(training_data))
# eval_dataset = Dataset.from_list(preprocess_data(eval_data))

# # Configurar los argumentos del entrenamiento
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./fine_tuned_t5_sql",
#     per_device_train_batch_size=4,  # Tamaño del lote reducido si tienes poca memoria GPU
#     num_train_epochs=20,
#     save_steps=500,
#     logging_dir="./logs",
#     logging_steps=10,
#     eval_steps=10,  # Usar eval_steps en lugar de evaluation_strategy (descontinuado)
#     save_strategy="epoch",        # Guardar al final de cada época
#     predict_with_generate=True,   # Activar generación en evaluación
#     generation_max_length=128,    # Longitud máxima para la generación
#     generation_num_beams=7,       # Número de beams para búsqueda
#     learning_rate=10e-5,
#     warmup_steps=500
# )

# # Usar un DataCollator para manejar el padding dinámico
# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# # Función para calcular métricas
# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Eliminar padding y limpiar textos
#     decoded_preds = [pred.strip() for pred in decoded_preds]
#     decoded_labels = [label.strip() for label in decoded_labels]

#     # Calcular exactitud
#     exact_matches = sum([pred == label for pred, label in zip(decoded_preds, decoded_labels)])
#     accuracy = exact_matches / len(decoded_preds) if len(decoded_preds) > 0 else 0

#     return {"accuracy": accuracy}

# # Configurar el Seq2SeqTrainer
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# # Entrenar el modelo
# trainer.train()

# # Guardar el modelo ajustado
# model.save_pretrained("./fine_tuned_t5_sql")
# tokenizer.save_pretrained("./fine_tuned_t5_sql")

# # Función para generar consultas SQL
# def generate_sql(question):
#     input_ids = tokenizer.encode(question, return_tensors="pt", max_length=128, truncation=True)
#     if torch.cuda.is_available():
#         input_ids = input_ids.to("cuda")
#         model.to("cuda")
#     output = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
#     sql_query = tokenizer.decode(output[0], skip_special_tokens=True)
#     return sql_query

# # Prueba con una pregunta
# question = "how many products are in total?"
# sql_query = generate_sql(question)
# print("Generated SQL:", sql_query)




















from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch

# Cargar el modelo y el tokenizador preentrenado
model_name = "cssupport/t5-small-awesome-text-to-sql"
model = T5ForConditionalGeneration.from_pretrained(model_name)
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
training_data = [
    {"input": "What are the details of all products?", 
     "output": "SELECT * FROM productos;"},
    {"input": "Show me the names of products with a price greater than 50", 
     "output": "SELECT nombre FROM productos WHERE precio > 50;"},
    {"input": "List the products added in the last 7 days", 
     "output": "SELECT * FROM productos WHERE fecha_ingreso >= NOW() - INTERVAL 7 DAY;"},
    {"input": "What is the price of the product called 'Smartphone'?", 
     "output": "SELECT precio FROM productos WHERE nombre = 'Smartphone';"},
    {"input": "Show me all products ordered by name in ascending order", 
     "output": "SELECT * FROM productos ORDER BY nombre ASC;"},
    {"input": "How many products have been added this month?", 
     "output": "SELECT COUNT(*) FROM productos WHERE MONTH(fecha_ingreso) = MONTH(CURDATE()) AND YEAR(fecha_ingreso) = YEAR(CURDATE());"},
    {"input": "Which products cost less than 100?", 
     "output": "SELECT * FROM productos WHERE precio < 100;"},
    {"input": "List the names of all products and their prices", 
     "output": "SELECT nombre, precio FROM productos;"},
    {"input": "Get the details of the product with ID 10", 
     "output": "SELECT * FROM productos WHERE id_producto = 10;"},
    {"input": "Show me the names of products added today", 
     "output": "SELECT nombre FROM productos WHERE DATE(fecha_ingreso) = CURDATE();"},
    {"input": "How many products have a price greater than 200?", 
     "output": "SELECT COUNT(*) FROM productos WHERE precio > 200;"},
    {"input": "What is the average price of all products?", 
     "output": "SELECT AVG(precio) FROM productos;"},
    {"input": "Show the most expensive product", 
     "output": "SELECT * FROM productos ORDER BY precio DESC LIMIT 1;"},
    {"input": "List the products that have been added in the last 30 days", 
     "output": "SELECT * FROM productos WHERE fecha_ingreso >= NOW() - INTERVAL 30 DAY;"},
    {"input": "Get the total price of all products combined", 
     "output": "SELECT SUM(precio) FROM productos;"},
]


eval_data = [
    {"input": "How many products have a price less than 50?", 
     "output": "SELECT COUNT(*) FROM productos WHERE precio < 50;"},
    {"input": "Show me the names of the cheapest products", 
     "output": "SELECT nombre FROM productos ORDER BY precio ASC LIMIT 5;"},
    {"input": "Which products were added yesterday?", 
     "output": "SELECT * FROM productos WHERE DATE(fecha_ingreso) = CURDATE() - INTERVAL 1 DAY;"},
    {"input": "What is the name of the product with the highest price?", 
     "output": "SELECT nombre FROM productos ORDER BY precio DESC LIMIT 1;"},
    {"input": "List all products and their details added this year", 
     "output": "SELECT * FROM productos WHERE YEAR(fecha_ingreso) = YEAR(CURDATE());"},
    {"input": "What is the total number of products in the database?", 
     "output": "SELECT COUNT(*) FROM productos;"},
    {"input": "Show the names and prices of products priced between 50 and 150", 
     "output": "SELECT nombre, precio FROM productos WHERE precio BETWEEN 50 AND 150;"},
    {"input": "List all products sorted by date added in descending order", 
     "output": "SELECT * FROM productos ORDER BY fecha_ingreso DESC;"},
    {"input": "How many products have been added this week?", 
     "output": "SELECT COUNT(*) FROM productos WHERE WEEK(fecha_ingreso) = WEEK(CURDATE()) AND YEAR(fecha_ingreso) = YEAR(CURDATE());"},
    {"input": "What is the price of the product with ID 15?", 
     "output": "SELECT precio FROM productos WHERE id_producto = 15;"},
]


synthetic_data = [
    {"input": "Which products cost more than 100?", 
     "output": "SELECT * FROM productos WHERE precio > 100;"},
    {"input": "How many products cost less than 50", 
     "output": "SELECT COUNT(*) FROM productos WHERE precio < 50;"},
]

#training_data.extend(synthetic_data);

# Preprocesar datos
train_dataset = Dataset.from_list(preprocess_data(training_data, schema_context))
eval_dataset = Dataset.from_list(preprocess_data(eval_data, schema_context))

# Configurar los argumentos del entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_t5_v2",
    per_device_train_batch_size=4,  # Tamaño del lote reducido si tienes poca memoria GPU
    num_train_epochs=30,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    eval_steps=10,  # Usar eval_steps en lugar de evaluation_strategy (descontinuado)
    save_strategy="epoch",        # Guardar al final de cada época
    predict_with_generate=True,   # Activar generación en evaluación
    generation_max_length=128,    # Longitud máxima para la generación
    generation_num_beams=7,       # Número de beams para búsqueda
    learning_rate=10e-5,
    warmup_steps=500
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
# model.save_pretrained("./fine_tuned_t5_v2")
# tokenizer.save_pretrained("./fine_tuned_t5_v2")

tokenizer = T5Tokenizer.from_pretrained("./fine_tuned_t5_v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_t5_v2")
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
question = "Show the names of products with a price greater than 20"
sql_query = generate_sql(question)
print("Generated SQL:", sql_query)