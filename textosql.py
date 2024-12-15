from transformers import GPT2LMHeadModel, GPT2Tokenizer
from db_connection import connect_to_db
import mysql.connector

finetunedGPT = GPT2LMHeadModel.from_pretrained("rakeshkiriyath/gpt2Medium_text_to_sql")
finetunedTokenizer = GPT2Tokenizer.from_pretrained("rakeshkiriyath/gpt2Medium_text_to_sql")


def generate_sql(question, model, tokenizer, max_length=256):
    prompt = f"Translate the following English question to SQL: {question}"

    encoded_input = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    sql_output = decoded_output[len(prompt):].strip()

    return sql_output

def execute_sql(query):
    connection = connect_to_db()
    cursor = connection.cursor()

    try:
        cursor.execute(query)
        if query.strip().upper().startswith("SELECT"):
            results = cursor.fetchall()
            return results
        else:
            connection.commit()
            return f"Query executed successfully: {query}"
    except mysql.connector.Error as err:
        return f"Error: {err}"
    finally:
        cursor.close()
        connection.close()

questions = [
  "List all products with a price higher than 20"
]

for question in questions:

  sql_query = generate_sql(question, finetunedGPT, finetunedTokenizer)
  sql_query = sql_query.strip()
  if sql_query.startswith('.'):
    sql_query = sql_query[1:].strip()
  print(f"Query: {sql_query}\n")

  #result = execute_sql(sql_query)
  #print(f"Result: {result}\n")
