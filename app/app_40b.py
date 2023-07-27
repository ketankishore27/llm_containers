import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from flask import Flask, request, jsonify

model_id = "falcon-40b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    response_body = {}
    try:
        prompt = request.data.decode('utf-8')
        batch = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
        batch = batch.to('cuda:0')
        with torch.cuda.amp.autocast():
            output_tokens = model.generate(input_ids = batch.input_ids, max_new_tokens=200, temperature=0.7,
                top_p=0.7, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        response_body['execution_trial'] = 'success'
        response_body['generated_response'] = generated_text
        response_body['exception'] = None
        return jsonify(response_body)
    except Exception as e:
        response_body['execution_trial'] = 'error'
        response_body['generated_response'] = None
        response_body['exception'] = str(e)
        return jsonify(response_body)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)