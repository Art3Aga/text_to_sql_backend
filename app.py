# from flask import Flask, jsonify, request
# from flask_cors import CORS
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# app = Flask(__name__)
# CORS(app)

# finetunedGPT = GPT2LMHeadModel.from_pretrained("rakeshkiriyath/gpt2Medium_text_to_sql")
# finetunedTokenizer = GPT2Tokenizer.from_pretrained("rakeshkiriyath/gpt2Medium_text_to_sql")

# def generate_text_to_sql(query, model, tokenizer, max_length=256):
#     prompt = f"Translate the following English question to SQL: {query}"

#     input_tensor = tokenizer.encode(prompt, return_tensors='pt').to('cpu')

#     output = model.generate(input_tensor, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

#     decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Return only the SQL part (removing the input text)
#     sql_output = decoded_output[len(prompt):].strip()

#     return sql_output

# @app.route('/tuning', methods=['POST'])
# def tuning():

#     dataPost = request.get_json()
#     query = dataPost.get('query', 'All employees')
    
#     sql_result = generate_text_to_sql(query, finetunedGPT, finetunedTokenizer)

#     response = {
#         "result": sql_result
#     }
#     return jsonify(response), 200


# if __name__ == '__main__':
#     app.run(debug=True)





from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Carga el modelo y el tokenizer
finetunedGPT = GPT2LMHeadModel.from_pretrained("rakeshkiriyath/gpt2Medium_text_to_sql")
finetunedTokenizer = GPT2Tokenizer.from_pretrained("rakeshkiriyath/gpt2Medium_text_to_sql")

# Inicializa la aplicación Flask
app = Flask(__name__)
CORS(app)

# Función para generar texto SQL
def generate_text_to_sql(query, model, tokenizer, max_length=256):
    prompt = f"Translate the following English question to SQL: {query}"

    input_tensor = tokenizer.encode(prompt, return_tensors='pt').to('cpu')

    output = model.generate(input_tensor, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Retorna solo la parte SQL (elimina el texto del prompt)
    sql_output = decoded_output[len(prompt):].strip()

    return sql_output

# Define el endpoint
@app.route('/tuning', methods=['POST'])
def tuning():
    # Verifica si el cuerpo tiene JSON
    if not request.json or 'query' not in request.json:
        return jsonify({'error': 'Missing "query" in request body'}), 400

    query = request.json['query']

    try:
        # Genera la consulta SQL
        sql_result = generate_text_to_sql(query, finetunedGPT, finetunedTokenizer)

        # Devuelve la respuesta como JSON
        return jsonify({'result': sql_result}), 200

    except Exception as e:
        # Maneja errores
        return jsonify({'error': str(e)}), 500

# Ejecuta la aplicación
if __name__ == '__main__':
    app.run(debug=True)
