def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--data_root', type=str, required=True, help='Path to composite dataset')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the current training run')
    args = parser.parse_args()

    model_path= args.model_path
    data_root = args.data_root
    run_name = args.run_name

    #create logging dirs 
    json_log = {}
    checkpoint_dir = f"experiments/checkpoints/{run_name}"
    log_dir= f"experiments/logs/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # Load dataset with augmentation
    test_dataset = EFormerDataset(root_dir=data_root+'/test',
                                size=(224, 224),
                                p_flip=0.0)



    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=True, num_workers=8,
    pin_memory=True,
    persistent_workers=True  # optional
    )

    #Load trained model
    model = EFormer().to(device)
    state_dict= torch.load(model_path ,map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  

    criterion= torch.nn.BCELoss()


    print("Start evaluating: ", flush=True)    
    print(f"Model on device: {next(model.parameters()).device}", flush=True)
    
    results= evaluate(model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device)    
    
    #log metrics    
    print(f"Loss: {results['loss']:.4f} | MAD: {results['mad']*1e3:.3f} | "
    f"MSE: {results['mse']*1e3:.3f} | Grad: {results['grad']*1e-3:.3f} | Conn: {results['conn']*1e-3:.3f}", flush=True)
    
    #save results    
    #with open(f"{log_dir}/metrics_log.json", "w") as f:
    #    json.dump(json_log, f, indent=4)
    print(json_log)
            
    print("Evaluation finished.", flush=True)



if __name__=="__main__":
    import torch.multiprocessing as mp

    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    import json
    import os
    import argparse
    import torch 
    from torch.utils.data import DataLoader
    from data.dataset import EFormerDataset
    from models.eformer import EFormer
    from utils.metrics import *
    from utils.evaluation import evaluate
    
    main()