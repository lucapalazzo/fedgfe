import torch.nn as nn
import torch

def check_optimizer_params(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    only_trainable: bool = True,
    verbose: bool = True,
):
    """
    Confronta i parametri presenti nell'optimizer con quelli del modello.

    Args
    ----
    model : nn.Module
        Il modello da cui prendere i parametri di riferimento.
    optimizer : torch.optim.Optimizer
        L'istanza dell'optimizer da verificare.
    only_trainable : bool, default=True
        Se True considera solo i parametri con requires_grad=True.
    verbose : bool, default=True
        Se True stampa un report leggibile; la funzione restituisce comunque
        un dict con i risultati.

    Returns
    -------
    result : dict
        {
            "missing": set di nomi parametri presenti nel modello ma non nell'optimizer,
            "extra"  : set di nomi parametri presenti nell'optimizer ma non nel modello,
            "ok"     : bool   # True se i due insiemi coincidono
        }
    """

    # ---- raccogli i parametri del modello (id -> name) ---------------------
    model_params = {
        id(p): name
        for name, p in model.named_parameters()
        if (p.requires_grad or not only_trainable)
    }

    # ---- raccogli i parametri dall'optimizer --------------------------------
    optim_ids = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p is None:
                continue       # Safety check per eventuali None
            if (p.requires_grad or not only_trainable):
                optim_ids.add(id(p))

    model_ids = set(model_params.keys())

    missing_ids = model_ids - optim_ids          # nel modello ma non nell'opt
    extra_ids   = optim_ids - model_ids          # nell'opt ma non nel modello

    result = {
        "missing": {model_params[i] for i in missing_ids},
        "extra":   {f"<unnamed_{i}>" for i in extra_ids},
        "ok":      len(missing_ids) == 0 and len(extra_ids) == 0,
    }

    if verbose:
        if result["ok"]:
            print("✅  Tutti i parametri coincidono: modello ↔ optimizer.")
        else:
            if result["missing"]:
                print("⚠️  Parametri *mancanti* nell'optimizer:")
                for n in sorted(result["missing"]):
                    print("   •", n)
            if result["extra"]:
                print("⚠️  Parametri *estranei* trovati nell'optimizer:")
                for n in sorted(result["extra"]):
                    print("   •", n)

    return result

def print_model_gradients_status (
    model: nn.Module,
):
   for name, p in model.named_parameters():
    if not p.requires_grad:                        # parametro «congelato»
        print(f"{name:25s}  requires_grad = False")
    elif p.grad is None:                           # mai usato nel grafo
        print(f"{name:25s}  NO grad (None)")
    else:                                          # tutto ok
        print(f"{name:25s}  grad norm = {p.grad.norm():.4g}")

