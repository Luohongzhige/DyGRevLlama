from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from transformers import LlamaForCausalLM
LlamaForCausalLM.generate
import torch
from GPU_get import *
dividing_line_len = 50
model_dir, device, tokenizer, model = None, None, None, None
history = []
init_state = False

def stdout_message(message):
    dividing_line = "=" * dividing_line_len
    print(dividing_line + "\n" + message + "\n" + dividing_line)

def errout_message(message):
    dividing_line = "!" * dividing_line_len
    print(dividing_line + "\n" + message + "\n" + dividing_line)
    exit()

def clear_history():
    global history
    history = []

def init(dir_input="", device_input=None):
    global init_state
    if init_state:
        return
    init_state = True
    stdout_message("Setting up directory and device")
    global device, model_dir, tokenizer, model
    if dir_input == "":
        errout_message("Please provide model directory")
    model_dir = dir_input
    if device_input is None:
        device = get_gpu(0.32)
    else:
        device = device_input
    if device is None:
        errout_message("No GPU available")
    else:
        stdout_message(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype='auto', device_map=device)

    stdout_message(f"finished loading model in device: {device}")

def dialogue(input_text):
    prompt = input_text
    messages = [
        {'role': 'system', 'content': ''},
        {'role': 'user', 'content': prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_input = tokenizer([text], return_tensors='pt').to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,  
        pad_token_id=tokenizer.eos_token_id,  
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    history.append([input_text, response])

    return response

def dialogue_with_history(input_text):
    prompt = input_text
    messages = [
        {'role': 'system', 'content': ''}
    ]
    for i in range (len(history)):
        messages.append({'role': 'user', 'content': history[i][0]})
        messages.append({'role': 'assistant', 'content': history[i][1]})
    messages.append({'role': 'user', 'content': prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_input = tokenizer([text], return_tensors='pt').to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,  
        pad_token_id=tokenizer.eos_token_id,  
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    history.append([input_text, response])

    return response

def forward(input_text):
    
    prompt = input_text
    messages = [
        {'role': 'system', 'content': ''}
    ]
    for i in range (len(history)):
        messages.append({'role': 'user', 'content': history[i][0]})
        messages.append({'role': 'assistant', 'content': history[i][1]})
    messages.append({'role': 'user', 'content': prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_input = tokenizer([text], return_tensors='pt').to(device)
    attention_mask = torch.ones(model_input.input_ids.shape, dtype=torch.long, device=device)
    output = model.forward(
        model_input.input_ids,
        attention_mask=attention_mask,  
    )
    return output

def text2hidden_state(text):
    model_input = tokenizer([text], return_tensors='pt', add_special_tokens=True).to(device)
    # add eos
    model_input = torch.cat((model_input.input_ids, torch.tensor([tokenizer.eos_token_id]).to(device).unsqueeze(0)), dim=1)
    attention_mask = torch.ones(model_input.shape, dtype=torch.long, device=device)
    output = model(model_input, attention_mask=attention_mask, output_hidden_states=True)
    eos_token_id = tokenizer.eos_token_id
    eos_position = (model_input == eos_token_id).nonzero(as_tuple=True)[1].item()
    torch.cuda.empty_cache()
    return output.hidden_states[-1][:, eos_position, :].detach().float().cpu().numpy()