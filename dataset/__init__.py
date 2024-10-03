dataset_roots = {
    "MMSafety": "./data/MM-SafetyBench/",
    "SPA_VL_harm": "/home/yiding/Code/VLM-Safety/LLaVA/reject_sampling/data/SPA_VL",
    "SPA_VL_help": "/data/SPA_VL/",
    "AdvBench": "/data/AdvBench/",
    "FigStep": "/data/FigStep/",
    "Attack": "/data/attack/"
}

def build_dataset(dataset_name):
    if dataset_name == "MMSafety":
        from .MMSafety import MMSafetyBench
        dataset = MMSafetyBench(dataset_roots[dataset_name])
    elif dataset_name == "SPA_VL_harm":
        from .SPA_VL_harm import SPA_VL_harm
        dataset = SPA_VL_harm(dataset_roots[dataset_name])
    elif dataset_name == "SPA_VL_help":
        from .SPA_VL_help import SPA_VL_help
        dataset = SPA_VL_help(dataset_roots[dataset_name])
    elif dataset_name == "AdvBench":
        from .AdvBench import AdvBench
        dataset = AdvBench(dataset_roots[dataset_name])
    elif dataset_name == "FigStep":
        from .FigStep import FigStep
        dataset = FigStep(dataset_roots[dataset_name])
    elif dataset_name == "Attack":
        from .attack import Attack
        dataset = Attack(dataset_roots[dataset_name])
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return dataset.get_data()