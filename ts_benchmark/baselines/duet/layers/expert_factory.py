import torch.nn as nn
from copy import deepcopy

# Importiere die beiden Experten-Typen, die wir erstellen können
from .linear_pattern_extractor import Linear_extractor
from .esn.reservoir_expert import ReservoirExpert


def create_experts(config) -> nn.ModuleList:
    """
    Erstellt eine `nn.ModuleList` mit einer Mischung aus linearen und ESN-Experten.

    Diese Fabrik-Funktion liest die Konfiguration, um die Anzahl und die spezifischen
    Parameter für jeden Expertentyp zu bestimmen.

    Args:
        config: Das Konfigurationsobjekt, das die folgenden Attribute enthalten muss:
                - num_linear_experts (int): Anzahl der zu erstellenden linearen Experten.
                - num_esn_experts (int): Anzahl der zu erstellenden ESN-Experten.
                - esn_configs (list, optional): Eine Liste von Dictionaries, wobei jedes
                  Dict spezifische Hyperparameter für einen ESN-Experten enthält.
                  Wenn nicht vorhanden, werden die globalen ESN-Parameter aus der
                  `config` verwendet.

    Returns:
        nn.ModuleList: Eine Liste, die die instanziierten Experten-Module enthält.
    """
    experts = nn.ModuleList()

    # 1. Erstelle die linearen Experten
    for _ in range(getattr(config, 'num_linear_experts', 0)):
        experts.append(Linear_extractor(config))

    # 2. Erstelle die ESN-Experten
    num_esn = getattr(config, 'num_esn_experts', 0)
    # Robusterer Check: Sicherstellen, dass es eine Liste ist.
    esn_configs = getattr(config, 'esn_configs', None)

    for i in range(num_esn):
        # Erstelle eine tiefe Kopie der Basis-Konfiguration, um sie nicht zu verändern
        expert_config = deepcopy(config)
        
        # Wenn eine spezifische Konfiguration für diesen Experten existiert, überschreibe die Basis-Werte
        if esn_configs and i < len(esn_configs):
            for key, value in esn_configs[i].items():
                setattr(expert_config, key, value)
        
        experts.append(ReservoirExpert(expert_config))

    return experts