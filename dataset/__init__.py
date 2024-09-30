dataset_roots = {
    "MMSafety": "YOUR_DATA_PATH",
    "SPA_VL_harm": "YOUR_DATA_PATH",
    "SPA_VL_help": "YOUR_DATA_PATH",
    "AdvBench": "YOUR_DATA_PATH",
    "FigStep": "YOUR_DATA_PATH",
    "MMSafety1": "YOUR_DATA_PATH",
    "Attack": "YOUR_DATA_PATH"
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