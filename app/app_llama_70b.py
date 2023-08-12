from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
from flask import Flask, request, jsonify

model_name_or_path = "llama2_70b_chat_uncensored-GPTQ"

use_triton = False
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        use_safetensors=True,
        trust_remote_code=False,
        max_memory = {0: "13GIB", 1: "13GIB", 2: "13GIB", 0: "13GIB"},
        inject_fused_attention=False,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
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
