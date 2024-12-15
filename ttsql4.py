from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, RobertaTokenizer
from datasets import Dataset

# Cargar modelo y tokenizador
model_name = "Salesforce/codet5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)


training_data = [
    {
        "input": "Show me the names of products with a price greater than 20",
        "output": "SELECT nombre FROM productos WHERE precio > 20;"
    },
    {
        "input": "products with a price less than 20",
        "output": "SELECT nombre FROM productos WHERE precio < 20;"
    },
    {
        "input": "List of products that entered on November 27",
        "output": "SELECT * FROM productos WHERE DATE(fecha_ingreso) = '2024-11-27';"
    },
    {
        "input": "List of products that arrived today",
        "output": "SELECT * FROM productos WHERE DATE(fecha_ingreso) = CURDATE();"
    },
    {
        "input": "How many products are in stock?",
        "output": "SELECT COUNT(*) FROM productos;"
    },
    {
        "input": "How many products are there in total?",
        "output": "SELECT COUNT(*) FROM productos;"
    },
    {
        "input": "Total amount of products",
        "output": "SELECT COUNT(*) FROM productos;"
    }
]

# Preparar los datos para el entrenamiento
dataset = Dataset.from_list([
    {
        "input_ids": tokenizer.encode(entry["input"], truncation=True, padding="max_length", max_length=128),
        "labels": tokenizer.encode(entry["output"], truncation=True, padding="max_length", max_length=128)
    }
    for entry in training_data
])

# Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_steps=100,
)

# Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Realizar el fine-tuning
trainer.train()
model.save_pretrained("./fine_tuned_codet5")
tokenizer.save_pretrained("./fine_tuned_codet5")

# Cargar el modelo ajustado
model = T5ForConditionalGeneration.from_pretrained("./fine_tuned_codet5")
tokenizer = RobertaTokenizer.from_pretrained("./fine_tuned_codet5")

def generate_sql(question):
    prefixed_question = f"SQL query: {question}"
    input_ids = tokenizer.encode(prefixed_question, return_tensors="pt", max_length=128, truncation=True)
    output = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    sql = tokenizer.decode(output[0], skip_special_tokens=True)
    return sql

# Prueba con una consulta
question = "Show me the names of all products."
sql_query = generate_sql(question)
print("Generated SQL:", sql_query)

