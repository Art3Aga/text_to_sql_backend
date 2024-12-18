# Consultas SQL mediante Inteligencia Artificial

# 📑 Tabla de Contenidos

1. [Objetivo](#objetivo)
2. [Lineamientos Generales](#lineamientos-generales)
3. [Finetuning del Modelo T5 para Generación de Consultas SQL](#finetuning-del-modelo-t5-para-generación-de-consultas-sql)
4. [Análisis de Entrenamiento del Modelo (Datos en Inglés)](#análisis-de-entrenamiento-del-modelo-datos-en-inglés)
5. [Análisis de Resultados con Dataset Español e Ingles](#análisis-de-resultados-con-dataset-español-e-ingles)
6. [Uso del modelo con fine-tuning para generar un endpoint](#uso-del-modelo-con-fine-tuning-para-generar-un-endpoint)
7. [Integración con Angular](#integración-con-angular)

# Traducción de lenguaje natural (inglés, español) a consultas SQL

## Detalles del requerimiento

### Objetivo

Desarrollar una aplicación web que permita realizar consultas a una base de datos utilizando el lenguaje natural (Inglés o Español) mediante el uso de la Inteligencia Artificial.
De esta manera aprender SQL de una forma más intuitiva para las personas nuevas en el entorno SQL.

Este prototipo a su vez, servirá como base para saber como realizar fine-tuning o entrenamiento de modelos de inteligencia artificial usando python.

**Descripción**

- Integración de modelo de inteligencia artificial basado en T5
- Desarrollar funcionalidad que permita al modelo de IA no solo entender el prompt en inglés, sino también realizar la traducción al español.
- Desarrollar funcionalidad que permita al usuario realizar la consulta (inglés o español), consumir la API del modelo IA, procesar la data y presentarla en 2 formatos: Lenguaje SQL y el resultado de la data consultada.



**¿Qué es HuggingFace?**

Hugging Face es una plataforma y comunidad de aprendizaje automático (Machine Learning) y ciencia de datos que permite:

- Crear, implementar y entrenar modelos de aprendizaje automático 
- Demostrar, ejecutar e implementar inteligencia artificial (IA) en aplicaciones en vivo 
- Alojar modelos y explorar conjuntos de datos para entrenarlos


**¿Porqué usar HuggingFace?**

A Hugging Face se le suele llamar el GitHub del aprendizaje automático porque permite a los desarrolladores compartir y probar su trabajo abiertamente . Hugging Face es conocido por su biblioteca Python Transformers, que simplifica el proceso de descarga y entrenamiento de modelos de aprendizaje automático, aparte del costo, ya que podemos usar los modelos de manera gratuita a diferencia de OPENAI, sin embargo, también ofrece servicios de pago por el uso de entrenamiento en la nube, si el caso es que no queremos hacerlo de manera local en nuestra computadora.


### Lineamientos Generales

**Acceso a HuggingFace**

No es necesario registrarse en la plataforma de huggingface solamente si deseamos utilizar los modelos y entrenarlos de manera local en nuestra computadora y ambiente, de lo contrario es necesario registrarse en la plataforma si queremos hacer uso de los servicios que ofrecen tales como el entrenamiento en la nube, etc.



# Finetuning del Modelo T5 para Generación de Consultas SQL

El fine-tuning es una técnica de aprendizaje automático e inteligencia artificial (IA) que consiste en adaptar un modelo preentrenado para que se ajuste a tareas o casos de uso específicos. 

Para realizar el fine-tuning, se vuelve a entrenar el modelo con un conjunto de datos más pequeño y específico. El modelo ajusta sus parámetros y sus incrustaciones para adaptarse al nuevo conjunto de datos. 

El objetivo del fine-tuning es mantener las capacidades originales del modelo y adaptarlo para que se ajuste a casos de uso más especializados

A continuación se detallará el proceso que se realizó de fine-tuning del modelo `cssupport/t5-small-awesome-text-to-sql` utilizando Hugging Face Transformers. Detallaremos las herramientas, librerías necesarias y una descripción de las principales configuraciones del proceso, incluyendo el análisis de los resultados obtenidos durante el entrenamiento.

---

## Requisitos Previos

### Herramientas y Librerías

- **Python 3.8+**
- **Transformers**: `pip install transformers`
- **Datasets**: `pip install datasets`
- **Torch**: `pip install torch`
- **SentencePiece**: `pip install sentencepiece`
- **ProtoBuf**: `pip install protobuf`
- **CUDA** (opcional, para entrenamiento en GPU)
- Archivos JSON que serviran como datasets o conjunto de datos para entrenar el modelo: `dataset.json` y `dataset_spanish.json`.
- **Buenos recursos del equipo**: para este ejemplo utilizamos una laptop con las siguientes caracteristicas: 
  - Procesador: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz   1.69 GHz
  - RAM: 64 GB
  - Almacenamiento: 2TB SSD

### Preparativos
1. Instalar las librerías mencionadas.
2. Preparar los datos de entrenamiento en formato JSON.
3. Asegurarse de tener suficiente espacio en disco para guardar los modelos ajustados, ya que al aplicarles el fine-tuning tienden a crecer en tamaño, en este caso llegando a 1.58 GB.

---

## Código Principal

### Cargar Modelo y Tokenizador
```python
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer

model_name = "cssupport/t5-small-awesome-text-to-sql"
config = T5Config.from_pretrained(model_name, dropout_rate=0.1)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
```

### Preprocesamiento de Datos
```python
def preprocess_data(data, schema_context):
    tokenized_data = []
    for entry in data:
        input_text = f"{schema_context}\nQuestion: {entry['input']}"
        output_text = entry["output"]
        inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=256)
        outputs = tokenizer(output_text, truncation=True, padding="max_length", max_length=128)
        labels = outputs["input_ids"]
        labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]
        tokenized_data.append({
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        })
    return tokenized_data
```

### Configuración de `Seq2SeqTrainingArguments`
El objeto Seq2SeqTrainingArguments de la librería Hugging Face establece los parámetros esenciales para entrenar el modelo:
- output_dir: el directorio donde se guardará nuestro modelo ya entrenado, en este caso lo guardaremos en la raiz del proyecto y se llamará "fine_tuned_t5_spanish".
- per_device_train_batch_size: indica el tamaño del lote para el entrenamiento en la cpu, en este caso, 4 significa que se procesarán 4 muestras por cada GPU/CPU en cada paso del entrenamiento.
- num_train_epochs: 50 indica que el modelo verá todos los datos de entrenamiento 50 veces.
- save_steps: Define cada cuántos pasos de entrenamiento se guardará un checkpoint del modelo, 500 significa que se guardará un modelo después de cada 500 pasos.
- save_strategy: "epoch" indica que los checkpoints se guardarán al final de cada época
- predict_with_generate: True habilita la generación de texto con el modelo.
- generation_max_length: 128 significa que las predicciones generadas tendrán como máximo 128 tokens
- generation_num_beams: 8 indica que se considerarán 8 secuencias candidatas al generar texto, mejorando la calidad de las predicciones
- learning_rate: Tasa de aprendizaje inicial para el optimizador, 2e-5 (o 0.00002) es un valor común para modelos preentrenados como T5
- warmup_steps: Número de pasos iniciales durante los cuales la tasa de aprendizaje aumenta linealmente desde 0 hasta su valor máximo, 135 significa que durante los primeros 135 pasos, la tasa de aprendizaje irá creciendo gradualmente.
- weight_decay: Factor de decaimiento de los pesos (regularización) aplicado para prevenir el sobreajuste., 0.01 es un valor típico que penaliza los pesos grandes

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./fine_tuned_t5_spanish",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    eval_steps=10,
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=8,
    learning_rate=2e-5,
    warmup_steps=135,
    weight_decay=0.01,
    save_total_limit=2
)
```

### Métricas de Evaluación con `compute_metrics`
La función `compute_metrics` evalúa las predicciones generadas durante el entrenamiento y cálculo de exactitud.

```python
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    exact_matches = sum([pred == label for pred, label in zip(decoded_preds, decoded_labels)])
    accuracy = exact_matches / len(decoded_preds) if len(decoded_preds) > 0 else 0
    return {"accuracy": accuracy}
```

### Configurar y Ejecutar `Seq2SeqTrainer`
```python
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

## Generar Consultas SQL con `generate_sql`
Una vez entrenado el modelo, podemos utilizarlo para generar consultas SQL:

```python
def generate_sql(question):
    input_text = f"{schema_context}\nQuestion: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        model.to("cuda")
    output = model.generate(input_ids, max_length=128, num_beams=7, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)
```

### Ejemplo
```python
question = "How many products have a price greater than 200?"
sql_query = generate_sql(question)
print("Generated SQL:", sql_query)
```

Ejecutamos el codigo con

```bash
python texttosql_t5_spanish.py
```

⚠️ **La duración del proceso de entrenamiento dependerá de los recursos de nuestro equipo, para el ejemplo tuvo una duración de 45 minutos de entrenamiento.**
![alt text](image.png)

---

# Análisis de Entrenamiento del Modelo (Datos en Inglés)

### Observaciones Clave:
- **Reducción de Pérdida:** Se observa una disminución significativa en la pérdida conforme avanzan las épocas, lo que indica que el modelo está aprendiendo de manera efectiva.
- **Norma del Gradiente:** Los valores de `grad_norm` presentan fluctuaciones, indicando ajustes en los parámetros del modelo.
- **Tasa de Aprendizaje:** La tasa de aprendizaje aumenta inicialmente hasta alcanzar un valor máximo y luego disminuye progresivamente.

---

## Análisis Detallado

### Pérdida (`loss`)
- **Tendencia General:** Disminuye de un valor inicial de ~3.02 a ~0.33.
- **Patrones:** 
  - Las primeras épocas muestran fluctuaciones significativas en la pérdida.
  - A partir de la época 10, la pérdida disminuye de manera más estable.
  - La pérdida final en las últimas épocas indica que el modelo podría estar acercándose a su límite de aprendizaje.

### Norma del Gradiente (`grad_norm`)
- **Tendencia General:** 
  - Valores altos (>10) en las primeras épocas, asociados a mayores ajustes en los pesos.
  - Gradualmente, los valores oscilan entre 2 y 4, indicando ajustes más precisos y menores oscilaciones en los pesos.
- **Valores Extremos:**
  - Máximo: 12.13 (época 4.29).
  - Mínimo: 1.41 (época 30.71).

### Tasa de Aprendizaje (`learning_rate`)
- **Evolución:** Sigue un patrón de incremento inicial hasta un valor pico y posteriormente disminuye:
  - Máximo: ~1.98e-05 (época 10).
  - Mínimo: ~2.12e-06 (época 45.71).
- **Eficiencia:** Este comportamiento asegura que el modelo explore adecuadamente al inicio y se ajuste con mayor precisión en etapas avanzadas.

---

## Tabla Resumida de Resultados

| Época  | Pérdida (`loss`) | Norma Gradiente (`grad_norm`) | Tasa de Aprendizaje (`learning_rate`) |
|--------|------------------|------------------------------|---------------------------------------|
| 0.71   | 3.0157           | 10.65                       | 1.48e-06                              |
| 10.0   | 1.6053           | 3.10                        | 1.98e-05                              |
| 20.0   | 1.0269           | 3.37                        | 1.48e-05                              |
| 30.0   | 0.4897           | 1.62                        | 9.91e-06                              |
| 40.0   | 0.4116           | 1.62                        | 4.95e-06                              |
| 45.71  | 0.4566           | 2.90                        | 2.12e-06                              |

---

## Conclusiones
1. **Efectividad del Entrenamiento:**
   - El modelo muestra una clara tendencia de mejora en las métricas de pérdida, lo cual indica un aprendizaje efectivo, sin embargo, se notó hay fluctuaciones significativas en `grad_norm` por lo que podría ser necesarios ajustes adicionales, como el uso de técnicas de regularización.

2. **Tasa de Aprendizaje:**
   - La estrategia de ajuste dinámico parece funcionar bien, ya que permite un balance entre exploración inicial y convergencia final.


---

# Análisis de Resultados con Dataset Español e Ingles
Durante el entrenamiento, el modelo registra los siguientes valores clave:

- **`loss`**: Indica qué tan bien el modelo se ajusta a los datos de entrenamiento. Una disminución constante sugiere convergencia.
- **`grad_norm`**: Refleja la magnitud de los gradientes. Valores excesivamente altos pueden sugerir inestabilidad en el entrenamiento.
- **`learning_rate`**: Controla el ajuste de los pesos durante el entrenamiento. Se ajusta automáticamente durante el calentamiento (`warmup_steps`).
- **`epoch`**: Representa el número de pasadas completas sobre los datos de entrenamiento.

1. **Pérdida (Loss):**
- Tendencia general:

  - La pérdida inicial es alta (3.189), lo cual es esperado en las primeras etapas del entrenamiento. A medida que avanzan las épocas, la pérdida disminuye consistentemente hasta alcanzar valores cercanos a 0.4 hacia la época 25.
  
  - La reducción de la pérdida sugiere que el modelo está aprendiendo a ajustarse al dataset combinado, lo cual es una buena señal.
  
- Oscilaciones:

  - Se observan algunas fluctuaciones menores en la pérdida (por ejemplo, entre las épocas 1.67 y 2.5 o entre las épocas 12.92 y 14.17). Estas podrían deberse a:
  
  - Diversidad lingüística en el dataset.
  - Tamaño de batch o configuraciones del optimizador.
2. **Norma del Gradiente (grad_norm):**
- La norma del gradiente comienza alta (~8.6) y disminuye gradualmente conforme avanza el entrenamiento, lo cual indica que los pasos del gradiente se vuelven más pequeños y estables.

- En algunas épocas (por ejemplo, 2.08 y 12.08), hay incrementos notables en la norma del gradiente, lo cual podría estar relacionado con:

  - Frases complejas o patrones nuevos en los datos.
  - Errores en el ajuste de hiperparámetros, como la tasa de aprendizaje.
- Indicador positivo:

  - En las últimas etapas, la norma del gradiente se estabiliza alrededor de valores bajos (~2 a 4), lo que sugiere que el modelo está convergiendo.
3. **Tasa de Aprendizaje (Learning Rate):**
- Se observa un esquema de ajuste progresivo:
  - La tasa de aprendizaje aumenta gradualmente hasta la época 5, y luego comienza a reducirse.
- Esto sigue un esquema común de "warmup-decay", que facilita un aprendizaje estable y evita oscilaciones grandes en la optimización.

1. **Conclusiones**
   - Observamos que el modelo en las primeras épocas la convinación de idiomas, le resulta un poco compleja, ya que este modelo no fue entrenado para entender español, solamente inglés, por lo que esto explica las fluctuaciones que tenemos en la pérdida y la norma del gradiente en etapas tempranas e intermedias, sin embargo, la pérdida decreciente y los valores estabilizados hacia el final indican que el modelo logra adaptarse al dataset combinado, por lo que la combinación de idiomas parece no haber afectado significativamente el aprendizaje.
   - Podemos concluir que al incluir ambos idiomas el modelo tiende a sentir complejo este cambio, por ello hay fluctuaciones en el gradiente con picos altos y bajos, sin embargo, si ampliamos el dataset y ajustamos parametros parametros de configuración, el modelo tendria resultados de mejor calidad.



# Uso del modelo con fine-tuning para generar un endpoint

Una vez entrenado el modelo, se guardará en la ruta que especificamos en la configuración, en este caso
en la raiz del proyecto "./fine_tuned_t5_spanish".

Creamos un nuevo archivo llamado "index.py" e instalamos las siguientes librerias:
```bash
pip install flask cors jsonify flask_cors mysql-connector-python
```
de esta manera vamos a usar la libreria de Flask para poder crear un servidor local y el endpoint que ejecutará a nuestro modelo entrenado.

Asi mismo la libreria de "mysql-connector-python" para conectarnos a nuestra base de datos de prueba, con el propósito de que una vez el modelo traduzca a SQL, podamos ejecutar dicha consulta directamente en 
nuestra base de datos MYSQL.

## Implementación

Importamos todas las librerias.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from db_connection import connect_to_db
import mysql.connector
```

Llamamos nuestro modelo entrenado

```python
output_dir = "./fine_tuned_t5_spanish"

model = T5ForConditionalGeneration.from_pretrained(output_dir)
tokenizer = T5Tokenizer.from_pretrained(output_dir)
```

Inicializamos la aplicación de flask

```python
app = Flask(__name__)
CORS(app)
```

**Haciendo uso del modelo entrenado**

Una vez hayamos llamado el modelo entrenado de nuestra carpeta, solo basta con definir nuevamente el esquema de la base de datos y definir los metodos "generate_sql" y "execute_sql"

```python
schema_context = """
Schema:
Table: productos
Columns:
  - id_producto (integer, primary key)
  - nombre (varchar(100))
  - precio (double)
  - fecha_ingreso (datetime)
"""

model.eval()

def generate_sql(question):
    input_text = f"{schema_context}\nQuestion: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        model.to("cuda")
    output = model.generate(input_ids, max_length=128, num_beams=7, early_stopping=True)
    sql_query = tokenizer.decode(output[0], skip_special_tokens=True)
    return sql_query


def execute_sql(query):
    connection = connect_to_db()
    cursor = connection.cursor()

    try:
        cursor.execute(query)
        if query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(zip(columns, row)) for row in results]
            print(columns)
            print(rows)
            return {"columns": columns, "rows": rows}  # Devuelve un diccionario
        else:
            connection.commit()
            return {"message": f"Query executed successfully: {query}"}
    except mysql.connector.Error as err:
        return {"error": f"Error: {err}"}
    finally:
        cursor.close()
        connection.close()
```

Por último creamos las rutas para ejecutar el modelo, en este caso se llamará "/tuning" y la otra ruta o endpoint será para ejecutar el SQL generado por el modelo directamente en la base de datos.

```python
@app.route('/tuning', methods=['POST'])
def tuning():
    if not request.json or 'query' not in request.json:
        return jsonify({'error': 'Missing "query" in request body'}), 400

    query = request.json['query']

    try:
        sql_result = generate_sql(query)
        return jsonify({'result': sql_result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/execute', methods=['POST'])
def execute():
    if not request.json or 'query' not in request.json:
        return jsonify({'error': 'Missing "query" in request body'}), 400

    query = request.json['query']

    try:
        sql_result = execute_sql(query)
        return jsonify(sql_result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

```

Corremos el servidor local

```bash
python index.py
```




# Integración con Angular

Una vez tengamos listo el servidor local de flask corriendo y listo nuestro endpoint, solo nos falta la parte del frontend, en este caso utilizaremos Angular en su version 19.0

Para ello, instalamos angular en nuestra computadora, asegurandonos de tener instalado Node (https://nodejs.org/es/download/package-manager), en este caso usaremos la v22.4.1


```bash
npm install -g @angular/cli@19.0.0
```

Luego, podemos clonar este proyecto en la carpeta "text_to_sql_frontend" y instalar todas las dependencias del proyecto de angular situandonos en la raiz del proyecto de angular, ejecutamos:

```bash
npm install
```

Una vez hecho esto podemos ejecutar el proyecto con el comando:

```bash
ng serve
```


Y listo, accedemos a la url que nos proporcionará el mismo angular en la linea de comandos, generalmente es la http://localhost:4200