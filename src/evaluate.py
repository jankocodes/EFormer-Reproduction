def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to composite dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory of logs/checkpoints')
    parser.add_argument('--run_name', type=str, required=True, help='Name of the current training run')
    parser.add_argument('--use_sa', type= lambda x: str(x).lower()=="true", default=True, help='Use self-attention layers')
    parser.add_argument('--use_ca', type= lambda x: str(x).lower()=="true", default=True, help='Use cross-attention layers')
    parser.add_argument('--first_upsampling', type= str, default='bilinear', choices=['bilinear', 'transconv'], help='First upsampling method')
    parser.add_argument('--second_upsampling', type= str, default='transconv', choices=['bilinear', 'transconv'], help='Second upsampling method')
    args = parser.parse_args()

    data_root = args.data_root
    model_path= args.model_path
    out_dir= args.out_dir
    run_name = args.run_name
    use_sa= args.use_sa
    use_ca= args.use_ca
    first_upsampling= args.first_upsampling
    second_upsampling= args.second_upsampling

    #create logging dirs 
    json_log = {}

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
    model = EFormer(use_sa=use_sa,
                    use_ca=use_ca,
                    first_upsampling=first_upsampling,
                    second_upsampling=second_upsampling).to(device)
    
    state_dict= torch.load(model_path ,map_location=device)
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
    with open(f"{out_dir}/evaluation_metrics.json", "w") as f:
        json.dump(json_log, f, indent=4)
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