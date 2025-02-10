import torch

def generate_with_replace_cls_to_other_embedding(tokenizer, model, prompt, embedding, device, eos_token_id, use_embedding, max_legnth=512):

    with torch.no_grad():
        if use_embedding:
            prompt  += "In particular, we will give you the embedding of the comments you need to predict:"
            input_ids = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
            input_embeddings = model.get_input_embeddings()(input_ids)
            input_embeddings = torch.cat((input_embeddings, torch.tensor(embedding, dtype=torch.bfloat16).to(device).unsqueeze(0).unsqueeze(0)), dim=1)
            prompt_end = ".Based the information above, please give a review of your predictions with as much historical information and embedding as you can."
        else:
            input_ids = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
            input_embeddings = model.get_input_embeddings()(input_ids)
            prompt_end = "Based the information above, please give a review of your predictions with as much historical information as you can."
        input_ids_end = tokenizer(prompt_end, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
        input_embeddings_end = model.get_input_embeddings()(input_ids_end)
        input_embeddings = torch.cat((input_embeddings, input_embeddings_end), dim=1)
        outputs = model.generate(inputs_embeds=input_embeddings, pad_token_id=eos_token_id, do_sample=True, max_length=max_legnth + len(input_embeddings[0]), num_return_sequences=1, temperature=0.7)
        #ids -> text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text