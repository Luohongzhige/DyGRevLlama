import argparse, sys, torch, numpy, tqdm, json, time
sys.path.append('/root/')
import GPU_get as GPU_get
from path import *
import numpy as np
import exargs
from dataset import HistoryEmbeddingDataset
from search import generate_with_replace_cls_to_other_embedding
import mytgn
parser = argparse.ArgumentParser(description="defualt args")
parser.add_argument('--dataset', type=str, default='cloth')
args = parser.parse_args()
DATASET = args.dataset
import llama

with open(f'{DATA_PATH}/config/{DATASET}.json') as f:
    config = json.load(f)

args = exargs.init(args, config)

print(args)

args.end_time = int(time.mktime(time.strptime(args.end_time, args.time_format)))

if not args.skip.part1:
    cnt = 0
    asin, customer = {}, {}
    tot_data = []
    with open(f'{args.data_path}', 'r') as f:
        for i in tqdm.tqdm(range(args.total_data_size)):
        # for line in f:
            # try:
                # context = json.loads(line)
            # except json.JSONDecodeError as e:
                # print(f"in {cnt}, Unable to parse JSON. Invalid text:{line}")
            # context = json.loads(line)
            context = json.loads(f.readline())
            data = {}
            for key, value in context.items():
                if key in args.need_item:
                    data[key] = value
            data[args.need_item[2]] = int(time.mktime(time.strptime(data[args.need_item[2]], args.time_format)))
            if args.need_item[3] not in data:
                continue
            if data[args.need_item[1]] not in asin:
                asin[data[args.need_item[1]]] = 0
            if data[args.need_item[0]] not in customer:
                customer[data[args.need_item[0]]] = 0
            asin[data[args.need_item[1]]] += 1
            customer[data[args.need_item[0]]] += 1
            tot_data.append(data)
            cnt += 1
    print(cnt)

    tot_data = list({tuple(sorted(d.items())): d for d in tot_data}.values())

    for _ in tqdm.tqdm(range(args.iteration)):
        tmp_data = []
        for i in range(len(tot_data)):
            if asin[tot_data[i][args.need_item[1]]] < args.k_core or customer[tot_data[i][args.need_item[0]]] < args.k_core:
                continue
            tmp_data.append(tot_data[i])
        asin, customer = {}, {}
        tot_data = tmp_data
        for i in range(len(tot_data)):
            if tot_data[i][args.need_item[1]] not in asin:
                asin[tot_data[i][args.need_item[1]]] = 0
            if tot_data[i][args.need_item[0]] not in customer:
                customer[tot_data[i][args.need_item[0]]] = 0
            asin[tot_data[i][args.need_item[1]]] += 1
            customer[tot_data[i][args.need_item[0]]] += 1

    tot_data.sort(key=lambda x: x[args.need_item[2]])

else:
    tot_data = []
    with open(f'{DATA_PATH}/{args.save_path.pretreatment}', 'r') as f:
        for line in f:
            context = json.loads(line)
            tot_data.append(context)
            if len(tot_data) > 200000:
                break

args = exargs.add_key(args, 'part1', {'edge_num':len(tot_data)})

if args.save.all == 1 or args.save.part1 == 1:
    with open(f'{DATA_PATH}/{args.save_path.pretreatment}', 'w') as f:
        for i in range(len(tot_data)):
            f.write(json.dumps(tot_data[i])+'\n')

print('part 2')
if not args.skip.part2:
    item, cnt = {}, 0
    with open(f'{DATA_PATH}/{args.save_path.csv}', "w") as f:
        f.write('u,i,idx,ts\n')
        for i in range(len(tot_data)):
            if tot_data[i][args.need_item[1]] not in item:
                item[tot_data[i][args.need_item[1]]] = cnt
                cnt += 1
            if tot_data[i][args.need_item[0]] not in item:
                item[tot_data[i][args.need_item[0]]] = cnt
                cnt += 1
            f.write(f'{item[tot_data[i][args.need_item[0]]]},{item[tot_data[i][args.need_item[1]]]},{i},{tot_data[i][args.need_item[2]]}\n')
    cnt = max(cnt, args.split)
    if len(tot_data) > cnt:
        tot_data = tot_data[:cnt]
    args = exargs.add_key(args, 'part2', {'node_num':cnt})
    
else:
    max_idx = 0
    with open(f'{DATA_PATH}/{args.save_path.csv}', "r") as f:
        f.readline()
        for line in f:
            tmp = line.split(',')
            max_idx = max(max_idx, int(tmp[0]))
            max_idx = max(max_idx, int(tmp[1]))
    args = exargs.add_key(args, 'part2', {'node_num':max_idx})

print('part 3')
if not args.skip.part3:
    llama.init(dir_input=f"/root/autodl-tmp/Llama-3-8B")
    embedding = np.zeros((len(tot_data), args.embedding_size))
    for i in tqdm.tqdm(range(len(tot_data))):
        embedding[i] = llama.text2hidden_state(tot_data[i][args.need_item[3]])

else:
    if args.llm == "llama":
        # mmapped_data = np.load(f'{DATA_PATH}/{args.save_path.embedding}', mmap_mode='r')
        # embedding = mmapped_data[:200000]
        embedding = np.load(f'{DATA_PATH}/{args.save_path.embedding}')
    else:
        embedding = np.zeros((len(tot_data), args.embedding_size))

if args.save.all or args.save.part3:
    np.save(f'{DATA_PATH}/{args.save_path.embedding}', embedding)

if not args.skip.part4:
    mytgn.init(args.tgn)
    embedding = mytgn.load(args.tgn)
    np.save(f'{DATA_PATH}/{args.save_path.memory}', embedding)
else:
    embedding = np.load(f'{DATA_PATH}/{args.save_path.memory}')
    print(embedding.shape)
    
if not args.skip.part5:
    if args.llm == "llama":
        score = []
        llama.init(dir_input=f"/root/autodl-tmp/Llama-3-8B")
        llama.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        dataset = HistoryEmbeddingDataset()
        source_history, destination_history = {}, {}
        for i in tqdm.tqdm(range(len(tot_data) - 200)):
            if i == 190000:
                break
            if tot_data[i][args.need_item[0]] not in source_history:
                source_history[tot_data[i][args.need_item[0]]] = []
            if tot_data[i][args.need_item[1]] not in destination_history:
                destination_history[tot_data[i][args.need_item[1]]] = []
            source_history[tot_data[i][args.need_item[0]]].append(tot_data[i][args.need_item[3]])
            destination_history[tot_data[i][args.need_item[1]]].append(tot_data[i][args.need_item[3]])
            if len(source_history[tot_data[i][args.need_item[0]]]) > args.history_len:
                source_history[tot_data[i][args.need_item[0]]].pop(0)
            if len(destination_history[tot_data[i][args.need_item[1]]]) > args.history_len:
                destination_history[tot_data[i][args.need_item[1]]].pop(0)
            if len(source_history[tot_data[i][args.need_item[0]]]) == args.history_len and len(destination_history[tot_data[i][args.need_item[1]]]) == args.history_len:
                dataset.append(source_history[tot_data[i][args.need_item[0]]], destination_history[tot_data[i][args.need_item[1]]], embedding[i], tot_data[i][args.need_item[3]])
                # score.append(tot_data[i][args.need_item[4]])
        # np.save(f'{DATA_PATH}/{args.save_path.score}', np.array(score))
        with open(f'{DATA_PATH}/{args.save_path.result}', 'w') as f:
            for i in tqdm.tqdm(range(6000, len(dataset))):
                prompt = "You are an expert in consumer psychology and are adept at predicting what a buyer's review will be when they buy an item through historical reviews of a particular purchaser and a particular item. Remember that historical customer reviews are about other products, and historical product reviews are from other customers. The history reviews will be give like this: buyers' history reviews text: \{text1\}.\\n\{text2\}.\\n\{text3\}.\\n\{text4\}.\\n\{text5\}.\\n.And historical reviews of items, it will be give like this: items' history reviews text: \{text1\}.\\n\{text2\}.\\n\{text3\}.\\n\{text4\}.\\n\{text5\}.\\n\n"
                prompt += "buyer's history reviews text:\n"
                for j in range(args.history_len):
                    prompt += '\{text' + dataset[i][0][j] + '\}.\n'
                prompt += "item's history reviews text:\n"
                for j in range(args.history_len):
                    prompt += '\{text' + dataset[i][1][j] + '\}.\n'
                f.write('ProMpt ' + prompt + '\n')
                f.write('*' * 100 + '\n')
                
                ans = generate_with_replace_cls_to_other_embedding(llama.tokenizer, llama.model, prompt, dataset[i][2], llama.device, llama.tokenizer.eos_token_id, True)
                f.write('mYModel ' + ans + '\n')
                f.write('*' * 100 + '\n')
                ans = generate_with_replace_cls_to_other_embedding(llama.tokenizer, llama.model, prompt, dataset[i][2], llama.device, llama.tokenizer.eos_token_id, False)
                f.write('BaSE ' + ans + '\n')
                f.write('*' * 100 + '\n')
                f.write('sTd ' + dataset[i][3] + '\n')
                f.write('-' * 100 + '\n')
    
    if args.llm == 'glm':
        device = GPU_get.get_gpu(0.2)
        from transformers import AutoTokenizer, AutoModel
        print("load model ", device)
        model = AutoModel.from_pretrained(f"{DATA_PATH}/glm-4-9b", trust_remote_code=True, device_map = device).half()
        print("load model 2")
        tokenizer = AutoTokenizer.from_pretrained(f"{DATA_PATH}/glm-4-9b", trust_remote_code=True)
        print("load model 3")
        dataset = HistoryEmbeddingDataset()
        source_history, destination_history = {}, {}
        for i in tqdm.tqdm(range(len(tot_data))):
            if tot_data[i][args.need_item[0]] not in source_history:
                source_history[tot_data[i][args.need_item[0]]] = []
            if tot_data[i][args.need_item[1]] not in destination_history:
                destination_history[tot_data[i][args.need_item[1]]] = []
            source_history[tot_data[i][args.need_item[0]]].append(tot_data[i][args.need_item[3]])
            destination_history[tot_data[i][args.need_item[1]]].append(tot_data[i][args.need_item[3]])
            if len(source_history[tot_data[i][args.need_item[0]]]) > args.history_len:
                source_history[tot_data[i][args.need_item[0]]].pop(0)
            if len(destination_history[tot_data[i][args.need_item[1]]]) > args.history_len:
                destination_history[tot_data[i][args.need_item[1]]].pop(0)
            if len(source_history[tot_data[i][args.need_item[0]]]) == args.history_len and len(destination_history[tot_data[i][args.need_item[1]]]) == args.history_len:
                dataset.append(source_history[tot_data[i][args.need_item[0]]], destination_history[tot_data[i][args.need_item[1]]], embedding[i], tot_data[i][args.need_item[3]])
        with open(f'{DATA_PATH}/{args.save_path.result}', 'w') as f:
            for i in tqdm.tqdm(range(len(dataset))):
                prompt = "You are an expert in consumer psychology and are adept at predicting what a buyer's review will be when they buy an item through historical reviews of a particular purchaser and a particular item. Remember that historical customer reviews are about other products, and historical product reviews are from other customers. The history reviews will be give like this: buyers' history reviews text: \{text1\}.\\n\{text2\}.\\n\{text3\}.\\n\{text4\}.\\n\{text5\}.\\n.And historical reviews of items, it will be give like this: items' history reviews text: \{text1\}.\\n\{text2\}.\\n\{text3\}.\\n\{text4\}.\\n\{text5\}.\\n\n"
                prompt += "buyer's history reviews text:\n"
                for j in range(args.history_len):
                    prompt += '\{text' + dataset[i][0][j] + '\}.\n'
                prompt += "item's history reviews text:\n"
                for j in range(args.history_len):
                    prompt += '\{text' + dataset[i][1][j] + '\}.\n'
                prompt_end = "Based the information above, please give a review of your predictions with as much historical information as you can."
                if len(prompt) > 30000:
                    continue
                prompt += prompt_end
                f.write('ProMpt ' + prompt + '\n')
                f.write('*' * 100 + '\n')
                #change prompt to embedding
                input_ids = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
                result =  model.generate(
                        input_ids,              
                        do_sample=True,  # 开启sample search
                        max_length=1024 + len(input_ids[0]),
                        temperature=0.7,  # 控制生成的随机性
                        top_k=5,  # Top-K sampling
                        top_p=5,  # Nucleus sampling
                        pad_token_id=tokenizer.eos_token_id
                    )
                generated_text = tokenizer.decode(result[0], skip_special_tokens=True)
                f.write('BaSE ' + generated_text + '\n')
                f.write('*' * 100 + '\n')
                f.write('sTd ' + dataset[i][3] + '\n')
                f.write('-' * 100 + '\n')