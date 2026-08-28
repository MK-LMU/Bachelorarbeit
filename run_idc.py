"""
Train IDC on the SAME dataset we use for SpEx, then extract its local gate matrix
and cluster labels -> saved to artifacts/idc_out/idc_out_<data><tag>_seed<k>.npz
for the comparison. The _seed suffix is what evaluate.py and train_all.py match
on; --tag and --outdir redirect the name.

We reuse IDC's real model + training loop (train_evaluate.BaseModule) unchanged;
we only inject a custom dataset class so both methods see identical inputs
(same X, same MinMaxScaler preprocessing IDC uses everywhere).
"""
import os, sys, argparse
import numpy as np

# NumPy-2 compat shim: IDC was written for NumPy<2 (uses removed aliases)
for _n, _v in [("infty", np.inf), ("NaN", np.nan), ("float", float),
               ("int", int), ("bool", bool), ("object", object)]:
    if not hasattr(np, _n):
        setattr(np, _n, _v)

IDC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IDC")
sys.path.insert(0, IDC_DIR)

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits, load_breast_cancer

import dataset as idc_dataset
from dataset import ClusteringDataset
import train_evaluate


def get_data(name):
    HERE = os.path.dirname(os.path.abspath(__file__))
    if name == "digits":
        d = load_digits(); return d.data.astype(float), d.target.astype(int), 10
    if name == "breast_cancer":
        d = load_breast_cancer(); return d.data.astype(float), d.target.astype(int), 2
    if name == "har":
        from wpaths import DATA
        d = np.load(os.path.join(DATA, "har_data.npz"))
        return d["X"].astype(float), d["y"].astype(int), 6
    if name == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(); return d.data.astype(float), d.target.astype(int), 3
    if name == "two_moons":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=2000, noise=0.12, random_state=0)
        return X.astype(float), y.astype(int), 2
    if name == "blobs":
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=2000, centers=5, n_features=2,
                          cluster_std=1.0, random_state=0)
        return X.astype(float), y.astype(int), 5
    if name == "cifar10":
        from wpaths import DATA
        d = np.load(os.path.join(DATA, "cifar_feats.npz"))
        return d["X"].astype(float), d["y"].astype(int), 10
    if name == "mnist_feats":
        from wpaths import DATA
        d = np.load(os.path.join(DATA, "mnist_feats.npz"))
        return d["X"].astype(float), d["y"].astype(int), 10
    raise ValueError(name)


def make_cfg(name, X, K, epochs, device):
    D = X.shape[1]
    bn = 64
    # Architecture variants for the spot check on whether network size, rather
    # than LGL/epochs, is the limit on Blobs/Digits — result: refuted, no smaller
    # network helps. (The script itself, check_architecture.py, is held back;
    # see README.) Without --arch EVERYTHING stays exactly as before — the
    # default is unchanged.
    arch = os.environ.get("ARCH", "")
    if arch:
        bn, hid, mid = {"small": (16, 64, 128), "tiny": (8, 16, 32)}[arch]
        cfg_arch = dict(gates_hidden_dim=hid,
                        encdec=[hid, hid, mid, bn, mid, hid, hid],
                        clustering_head=[bn, mid], aux_classifier=[mid])
    else:
        cfg_arch = dict(gates_hidden_dim=256,
                        encdec=[256, 256, 512, bn, 512, 256, 256],
                        clustering_head=[bn, 512], aux_classifier=[512])
    cfg = OmegaConf.create({
        "dataset": f"SHARED_{name}", "data_dir": ".", "scaler": "MinMaxScaler",
        "validate": True, "seeds": 1, "save_seed_checkpoints": False,
        "batch_size": min(256, X.shape[0]),
        "ae_non_gated_epochs": max(2, epochs // 20),
        "ae_pretrain_epochs": max(5, epochs // 3),
        "start_global_gates_training_on_epoch": int(epochs * 0.75),
        "mask_percentage": 0.5, "latent_noise_std": 0.01,
        "trainer": {"devices": 1, "accelerator": device, "max_epochs": epochs,
                    # Parts of IDC's encoder/GTCR path have no deterministic
                    # CUDA kernel, so Lightning would abort. seed_everything()
                    # still fixes init and shuffling, and retrains reproduce
                    # exactly -- see the sigma=0 control in
                    # perturbation_stability.py.
                    "deterministic": False, "logger": False,
                    "log_every_n_steps": 5, "check_val_every_n_epoch": max(1, epochs // 10),
                    "enable_checkpointing": False, "num_sanity_val_steps": 0,
                    "enable_progress_bar": False},
        "gtcr_loss": True, "gtcr_projection_dim": None, "gtcr_eps": 1, "eps": 0.1,
        "use_gating": True, **cfg_arch, "tau": 100,
        "local_gates_lambda": float(os.environ.get("LGL", 1.0)),
        "global_gates_lambda": 10, "gtcr_lambda": 0.01,
        "lr": {"pretrain": 1e-3, "clustering": 1e-3, "aux_classifier": 1e-1},
        "sched": {"pretrain_min_lr": 1e-4, "clustering_min_lr": 1e-4},
    })
    return cfg


def build_mnist(device, epochs):
    """Use IDC's OWN validated MNIST config + MNIST10K dataset class (784-D pixels,
    MinMaxScaler). Returns the cfg ONLY -- X/y/K come from model.train_dataset
    once BaseModule(cfg) has instantiated IDC's own MNIST loader."""
    HERE = os.path.dirname(os.path.abspath(__file__))
    cfg = OmegaConf.load(os.path.join(IDC_DIR, "cfg", "cfg_mnist.yaml"))
    from wpaths import DATA
    cfg.data_dir = os.path.join(DATA, "mnist_data")
    os.makedirs(cfg.data_dir, exist_ok=True)
    cfg.trainer.accelerator = device
    cfg.trainer.deterministic = False   # the one override of the paper cfg,
                                        # for the reason given in make_cfg
    cfg.trainer.logger = False
    cfg.trainer.enable_progress_bar = False
    cfg.seeds = 1
    if epochs:
        cfg.trainer.max_epochs = epochs
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="digits")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed; outputs get a _seed<k> suffix")
    ap.add_argument("--lgl", type=float, default=None,
                    help="override local_gates_lambda (gate penalty); the 2-D "
                         "synthetics collapse under the default 1.0")
    ap.add_argument("--tag", default="",
                    help="suffix for output files (e.g. _tuned), so tuned runs "
                         "never overwrite default-config runs")
    ap.add_argument("--outdir", default="",
                    help="write npz+model here instead of artifacts/idc_out|models "
                         "(used by tune_idc.py for throwaway grid candidates)")
    ap.add_argument("--perturb", default="",
                    help="RQ2 stability: perturb the data after loading, e.g. "
                         "noise:0.05:2 or subsample:0.9:1 (see perturbations.py)")
    ap.add_argument("--arch", default="", choices=["", "small", "tiny"],
                    help="shrink the network (spot check on whether the architecture "
                         "is the limit on Blobs/Digits); empty = the "
                         "unchanged default used for every reported run")
    args = ap.parse_args()
    if args.lgl is not None:
        os.environ["LGL"] = str(args.lgl)   # read by make_cfg
    if args.arch:
        os.environ["ARCH"] = args.arch      # read by make_cfg

    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()} -> accelerator={device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    perturb_idx = None
    if args.data == "mnist":
        cfg = build_mnist(device, args.epochs)      # IDC's real config + dataset
        seed_everything(args.seed); np.random.seed(args.seed)
        model = train_evaluate.BaseModule(cfg)
        X = np.asarray(model.train_dataset.data, dtype=np.float32)
        y = np.asarray(model.train_dataset.labels, dtype=int)
        K = int(model.train_dataset.num_clusters)
    else:
        Xraw, y, K = get_data(args.data)
        X = MinMaxScaler().fit_transform(Xraw).astype(np.float32)
        if args.perturb:
            from perturbations import perturb
            X, y, perturb_idx = perturb(X, y, args.perturb)
            X = np.ascontiguousarray(X, np.float32)
            print(f"perturbed data: {args.perturb} -> X={X.shape}")
        class _Shared(ClusteringDataset):
            @classmethod
            def setup(cls, cfg):
                return cls(X, y, K)
        setattr(idc_dataset, f"SHARED_{args.data}", _Shared)
        cfg = make_cfg(args.data, X, K, args.epochs or 120, device)
        seed_everything(args.seed); np.random.seed(args.seed)
        model = train_evaluate.BaseModule(cfg)

    trainer = Trainer(**cfg.trainer)
    trainer.fit(model)

    # extract IDC's local gate matrix + predicted labels on the full data
    model.eval()
    dev = next(model.parameters()).device
    Xt = torch.tensor(X.astype(np.float32), device=dev)
    with torch.no_grad():
        gates = model.gating_net.get_gates(Xt).cpu().numpy()   # (N, D) in [0,1]

    def idc_infer(Xmasked):
        """X(masked) -> hard cluster labels, IDC's own forward pass."""
        with torch.no_grad():
            xt = torch.tensor(np.asarray(Xmasked, np.float32), device=dev)
            g = model.gating_net.get_gates(xt)
            emb = model.encdec.encoder(xt * g)
            return model.clustering_head(emb).argmax(-1).cpu().numpy()

    labels_pred = idc_infer(X)
    print(f"IDC done. best ACC={model.best_acc:.3f} ARI={model.best_ari:.3f} "
          f"NMI={model.best_nmi:.3f}; gates shape {gates.shape}, "
          f"mean open gates/sample={(gates>0).sum(1).mean():.1f}/{X.shape[1]}")

    # compute IDC's interpretability metrics WHILE the model is alive (faithfulness
    # needs the forward pass) -- using the shared (=IDC's own) metric code
    from metrics import all_metrics, faithfulness_drop
    m = all_metrics("IDC", X, gates, idc_infer, y, K, skip_distance_metrics=True)
    # drop-based faithfulness: defined even where the correlation variant
    # degenerates (<2 used features, e.g. Breast Cancer)
    m.update(faithfulness_drop(gates, X, idc_infer, y, X.shape[1], K))
    print("IDC interpretability metrics:", {k: round(v, 4) for k, v in m.items()})

    from wpaths import MODELS, idc_out
    # save the trained weights so masked inference stays possible WITHOUT
    # retraining (the original runs never persisted the model)
    model_dir = args.outdir or MODELS
    ckpt = os.path.join(model_dir, f"idc_model_{args.data}{args.tag}_seed{args.seed}.pt")
    torch.save({"state_dict": model.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "seed": args.seed, "data": args.data, "lgl": args.lgl,
                "epochs": args.epochs}, ckpt)
    print("saved model:", ckpt)

    out = (os.path.join(args.outdir, f"idc_out_{args.data}{args.tag}_seed{args.seed}.npz")
           if args.outdir else
           idc_out(f"idc_out_{args.data}{args.tag}_seed{args.seed}.npz"))
    extra = {"perturb_idx": perturb_idx} if perturb_idx is not None else {}
    np.savez(out, X=X, gates=gates, labels_pred=labels_pred, y_true=y, K=K,
             seed=args.seed, **extra,
             config_validated=(args.data == "mnist"),  # only MNIST uses IDC's paper config
             acc=model.best_acc, ari=model.best_ari, nmi=model.best_nmi,
             faithfulness=m["faithfulness"], diversity=m["diversity"],
             uniqueness=m["uniqueness"], stability=m["stability"],
             generalizability=m["generalizability"],
             faithfulness_top1drop=m["faithfulness_top1drop"],
             faithfulness_aopc=m["faithfulness_aopc"],
             faithfulness_n_used=m["faithfulness_n_used"])
    print("saved:", out)


if __name__ == "__main__":
    main()
