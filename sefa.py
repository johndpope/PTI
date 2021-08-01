import argparse
import pickle
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sefa")
    parser.add_argument("--out", type=str, default="factor.pt", help="name of the result factor file")
    parser.add_argument("--ckpt", type=str, default="pretrained_models/webtoon001693.pkl", help="model checkpoint")
    # parser.add_argument("--ckpt", type=str, default="checkpoints/model_RKJBGNOQHLMZ_336944_15.pkl", help="model checkpoint")
    
    args = parser.parse_args()

    with open(args.ckpt, 'rb') as f:
        obj = f.read()
    ckpt = pickle.loads(obj, encoding='latin1')

    weights = []
    names = []

    for name in ckpt["G_ema"].state_dict():
        if ("synthesis" in name) and ("torgb" not in name) and ("affine.weight" in name):
            weight = ckpt["G_ema"].state_dict()[name]
            weights.append(weight)
            names.append(name)
            print(f"[INFO] append {len(names)}: {name}... {weight.shape}")
    # for name in ckpt.state_dict():
    #     if ("synthesis" in name) and ("torgb" not in name) and ("affine.weight" in name):
    #         weight = ckpt.state_dict()[name]
    #         weights.append(weight)
    #         names.append(name)
    #         print(f"[INFO] append {len(names)}: {name}... {weight.shape}")
    W = torch.cat(weights, 0)
    print(f"[INFO] concatenated W.shape: {W.shape}")

    eigvec = torch.svd(W).V.to("cpu")
    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)
    print(f"[INFO] saved successfully to {args.out}")
