import sys, logging, time, math, torch
sys.path.append('/home/qiaoy/')
from path import *
sys.path.append(f'{WORK_PATH}')
from GPU_get import *
from tgn.model.tgn import TGN
from tgn.utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from tgn.utils.data_processing import compute_time_statistics, get_data_node_classification
from tgn.evaluation.evaluation import eval_node_classification
import numpy as np

tgn, decoder = None, None


def init(args):
    
    global tgn, decoder

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    UNIFORM = args.uniform
    NUM_LAYER = args.n_layer
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    DATA = args.data
    
    create_folder(f'{DATA_PATH}/saved_models/')
    create_folder(f'{DATA_PATH}/saved_checkpoints/')
    MODEL_SAVE_PATH = f'./saved_models/{args.data}' + '\
    node-classification.pth'
    

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('{}/{}.log'.format(LOG_PATH, str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    full_data, node_features, edge_features, train_data, val_data, test_data = \
    get_data_node_classification(DATA_PATH, args, DATA, args.node_num, args.edge_num, args.shape, use_validation=args.use_validation)

    max_idx = max(full_data.unique_nodes)

    train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

    # Set device
    device = get_gpu(0.5)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    for i in range(args.n_runs):

        # Initialize Model
        tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                    edge_features=edge_features, device=device,
                    n_layers=NUM_LAYER,
                    n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                    message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                    memory_update_at_start=not args.memory_update_at_end,
                    embedding_module_type=args.embedding_module,
                    message_function=args.message_function,
                    aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
                    mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                    mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                    use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                    use_source_embedding_in_message=args.use_source_embedding_in_message)

        tgn = tgn.to(device)

        num_instance = len(train_data.sources)
        num_batch = math.ceil(num_instance / BATCH_SIZE)
        
        logger.debug('Num of training instances: {}'.format(num_instance))
        logger.debug('Num of batches per epoch: {}'.format(num_batch))

        tgn.eval()
        logger.info('TGN models loaded')
        logger.info('Start training node classification task')

        decoder = MLP(node_features.shape[1] * 2,node_features.shape[1],  drop=DROP_OUT, ablation = args.ablation)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        tgn_optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr)
        decoder = decoder.to(device)
        decoder_loss_criterion = torch.nn.MSELoss()

        val_aucs = []
        train_losses = []

        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for epoch in range(args.n_epoch):
            start_epoch = time.time()
            if USE_MEMORY:
                tgn.memory.__init_memory__()
            tgn = tgn.train()
            decoder = decoder.train()
            loss = 0
            for k in range(num_batch):
                s_idx = k * BATCH_SIZE
                e_idx = min(num_instance, s_idx + BATCH_SIZE)

                sources_batch = train_data.sources[s_idx: e_idx]
                destinations_batch = train_data.destinations[s_idx: e_idx]
                timestamps_batch = train_data.timestamps[s_idx: e_idx]
                edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
                labels_batch = train_data.labels[s_idx: e_idx]

                size = len(sources_batch)

                decoder_optimizer.zero_grad()
                with torch.no_grad():
                    source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,                                    destinations_batch,                                      destinations_batch,                                      timestamps_batch,                                        edge_idxs_batch,                                        NUM_NEIGHBORS)
                labels_batch_torch = torch.from_numpy(labels_batch).float().to(device)
                pred = decoder(torch.cat([source_embedding, destination_embedding], dim=1))
                decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
                decoder_loss.backward()
                decoder_optimizer.step()
                tgn_optimizer.step()
                loss += decoder_loss.item()
            train_losses.append(loss / num_batch)

            val_auc = eval_node_classification(tgn, decoder, val_data, full_data.edge_idxs, BATCH_SIZE,
                                            n_neighbors=NUM_NEIGHBORS)
            val_aucs.append(val_auc)

            logger.info(f'Epoch {epoch}: train loss: {loss / num_batch}, val auc: {val_auc}, time: {time.time() - start_epoch}')
        
        if args.use_validation:
            if early_stopper.early_stop_check(val_auc):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                torch.save(decoder.state_dict(), f'{DATA_PATH}/{args.save_path.decoder}')
                torch.save(tgn.state_dict(), f'{DATA_PATH}/{args.save_path.tgn}')
                break
            else:
                torch.save(decoder.state_dict(), f'{DATA_PATH}/{args.save_path.decoder}')
                torch.save(tgn.state_dict(), f'{DATA_PATH}/{args.save_path.tgn}')


def load(args):
    global tgn, decoder

    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    UNIFORM = args.uniform
    NUM_LAYER = args.n_layer
    USE_MEMORY = args.use_memory
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    DATA = args.data
    
    create_folder(f'{DATA_PATH}/saved_models/')
    create_folder(f'{DATA_PATH}/saved_checkpoints/')
    MODEL_SAVE_PATH = f'./saved_models/{args.data}' + '\
    node-classification.pth'
    

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('{}/{}.log'.format(LOG_PATH, str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    full_data, node_features, edge_features, train_data, val_data, test_data = \
    get_data_node_classification(DATA_PATH, args, DATA, args.node_num, args.edge_num, args.shape, use_validation=args.use_validation)

    max_idx = max(full_data.unique_nodes)

    train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM, max_node_idx=max_idx)

    # Set device
    device = get_gpu(1)

    # Compute time statistics
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

    
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
                edge_features=edge_features, device=device,
                n_layers=NUM_LAYER,
                n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
                message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                memory_update_at_start=not args.memory_update_at_end,
                embedding_module_type=args.embedding_module,
                message_function=args.message_function,
                aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
                mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
                mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
                use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                use_source_embedding_in_message=args.use_source_embedding_in_message)
    decoder = MLP(node_features.shape[1] * 2,node_features.shape[1],  drop=DROP_OUT, ablation = args.ablation)

    tgn.load_state_dict(torch.load(f'{DATA_PATH}/{args.save_path.tgn}', weights_only=True))
    decoder.load_state_dict(torch.load(f'{DATA_PATH}/{args.save_path.decoder}', weights_only=True))

    tgn.memory.__init_memory__()
    tgn = tgn.to(device)
    decoder = decoder.to(device)
    
    full_data, _, _, _, _, _ =get_data_node_classification(DATA_PATH, args, args.data, args.node_num, args.edge_num, args.shape, use_validation=args.use_validation)
    num_instance = len(full_data.sources)
    num_instance -= 200
    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    val_aucs = []
    result = []
    # get the models output for all data
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance, s_idx + BATCH_SIZE)
        # print(s_idx, e_idx)
        sources_batch = full_data.sources[s_idx: e_idx]
        destinations_batch = full_data.destinations[s_idx: e_idx]
        timestamps_batch = full_data.timestamps[s_idx: e_idx]
        edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
        labels_batch = full_data.labels[s_idx: e_idx]
        size = len(sources_batch)
        with torch.no_grad():
            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch, destinations_batch, destinations_batch, timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
            pred = decoder(torch.cat([source_embedding, destination_embedding], dim=1))
            val_aucs.append(pred.cpu().numpy())
            result.append(pred.cpu().numpy())
            print(pred.shape)
    result = np.concatenate(result)
    return result