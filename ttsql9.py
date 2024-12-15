from transformers import GPT2LMHeadModel, GPT2Tokenizer

finetunedGPT = GPT2LMHeadModel.from_pretrained("rakeshkiriyath/gpt2Medium_text_to_sql")
finetunedTokenizer = GPT2Tokenizer.from_pretrained("rakeshkiriyath/gpt2Medium_text_to_sql")

def generate_text_to_sql(query, model, tokenizer, max_length=256):
    prompt = f"Translate the following English question to SQL: {query}"

    input_tensor = tokenizer.encode(prompt, return_tensors='pt').to('cpu')

    output = model.generate(input_tensor, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return only the SQL part (removing the input text)
    sql_output = decoded_output[len(prompt):].strip()

    return sql_output

queryList = ["Find all employees who are under age 30."]

for query in queryList:

  sql_result = generate_text_to_sql(query, finetunedGPT, finetunedTokenizer)
  print(sql_result,"\n")
