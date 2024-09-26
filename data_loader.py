from dataset import build_dataset

def MM_SafetyBench_loader(batch_size:int):
    dataset, _ = build_dataset("MMSafety")
    return dataset

def Coco_loader(batch_size:int):
    dataset = build_dataset("coco")
    return dataset

def SPA_VL_harm_loader(batch_size:int):
    dataset = build_dataset("SPA_VL_harm")
    return dataset

def SPA_VL_help_loader(batch_size:int):
    dataset = build_dataset("SPA_VL_help")
    return dataset

def AdvBench_loader(batch_size:int):
    dataset = build_dataset("AdvBench")
    return dataset

def FigStep_loader(batch_size:int):
    dataset = build_dataset("FigStep")
    return dataset

def Attack_loader(batch_size:int):
    dataset = build_dataset("Attack")
    return dataset