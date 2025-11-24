def check_params_graph_vs_optimizer(model, optimizer, loss):
    # 1) pulisco i gradienti per non confondermi con roba vecchia
    optimizer.zero_grad()
    
    # 2) costruisco il grafo e i grad della loss
    loss.backward(retain_graph=True)
    
    # 3) mappa id(param) -> nome, per stampare qualcosa di leggibile
    id_to_name = {id(p): n for n, p in model.named_parameters()}
    
    # 4) parametri che sono effettivamente nel grafo (grad non None)
    in_graph = {id(p) for p in model.parameters() if p.grad is not None}
    
    # 5) parametri che l'optimizer proverà ad aggiornare
    in_optim = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    }
    
    # 6) insiemi di confronto
    only_in_optim = in_optim - in_graph   # param nell'optimizer ma fuori dal grafo
    only_in_graph = in_graph - in_optim   # param nel grafo ma non nell'optimizer
    in_both       = in_graph & in_optim   # param correttamente coperti
    
    print("=== PARAMETRI NEL GRAFO E NELL'OPTIMIZER (OK) ===")
    for pid in sorted(in_both, key=lambda x: id_to_name.get(x, "")):
        print(" -", id_to_name.get(pid, f"<unknown {pid}>"))
    
    print("\n=== PARAMETRI SOLO NELL'OPTIMIZER (grad = None, non aggiornati) ===")
    for pid in sorted(only_in_optim, key=lambda x: id_to_name.get(x, "")):
        print(" -", id_to_name.get(pid, f"<unknown {pid}>"))
    
    print("\n=== PARAMETRI SOLO NEL GRAFO (NON nell'optimizer, quindi mai aggiornati) ===")
    for pid in sorted(only_in_graph, key=lambda x: id_to_name.get(x, "")):
        print(" -", id_to_name.get(pid, f"<unknown {pid}>"))