from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import Dataset

# 1. Cargar el modelo y el tokenizador preentrenado
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# 2. Crear datos de entrenamiento con tu esquema de base de datos
training_data = [
    {"input": "List all products with a price greater than 20", 
     "output": "SELECT name FROM products WHERE price > 20;"},
    {"input": "Show details of products added today", 
     "output": "SELECT * FROM products WHERE DATE(added_date) = CURDATE();"},
    {"input": "How many products are currently in stock?", 
     "output": "SELECT COUNT(*) FROM products WHERE stock > 0;"},
    {"input": "What is the price of the product with ID 5?", 
     "output": "SELECT price FROM products WHERE product_id = 5;"},
    {"input": "List all products that cost less than 50", 
     "output": "SELECT * FROM products WHERE price < 50;"}
]

# Datos de evaluación
eval_data = [
    {"input": "Count the number of products with a price greater than 100", 
     "output": "SELECT COUNT(*) FROM products WHERE price > 100;"},
    {"input": "Show products added yesterday", 
     "output": "SELECT * FROM products WHERE DATE(added_date) = CURDATE() - INTERVAL 1 DAY;"}
]

# 3. Preprocesar los datos para convertirlos en formato compatible con Hugging Face
def preprocess_data(data):
    return [
        {
            "input_ids": tokenizer.encode(f"Translate to SQL: {item['input']}", truncation=True, padding="max_length", max_length=128),
            "labels": tokenizer.encode(item["output"], truncation=True, padding="max_length", max_length=128)
        }
        for item in data
    ]

train_dataset = Dataset.from_list(preprocess_data(training_data))
eval_dataset = Dataset.from_list(preprocess_data(eval_data))

# 4. Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./t5_sql_finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch"
)

# 5. Configurar el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 6. Entrenar el modelo
trainer.train()

# 7. Guardar el modelo ajustado
model.save_pretrained("./t5_sql_finetuned")
tokenizer.save_pretrained("./t5_sql_finetuned")

# 8. Función para generar consultas SQL personalizadas
def generate_sql(question, model, tokenizer):
    input_text = f"Translate to SQL: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128)
    output = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Ejemplo de uso
question = "List all products with a price greater than 20"
sql_query = generate_sql(question, model, tokenizer)
print("Generated SQL:", sql_query)
