from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from db_connection import connect_to_db
import mysql.connector

output_dir = "./fine_tuned_t5_spanish"

model = T5ForConditionalGeneration.from_pretrained(output_dir)
tokenizer = T5Tokenizer.from_pretrained(output_dir)

app = Flask(__name__)
CORS(app)

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
