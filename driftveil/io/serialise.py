import json
from typing import Any, Optional
import pandas as pd

def save_pact(pact: Any, path: str):
    """Serialize pact contracts and cached reference statistics to JSON."""
    serialized_contracts = []
    
    for contract in pact.contracts:
        # Convert tuple targets (like pairs) to lists for JSON compatibility
        target = contract["target"]
        if isinstance(target, tuple):
            target = list(target)
            
        serialized_contracts.append({
            "target_type": contract["target_type"],
            "target": target,
            "contract_type": contract["contract_type"],
            "params": contract["params"],
            "severity": contract["severity"],
            "ref_stats": contract["ref_stats"]
        })
        
    data = {
        "version": "0.1.0",
        "contracts": serialized_contracts
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_pact(path: str, reference_df: Optional[pd.DataFrame] = None) -> Any:
    """Load pact contracts and cached reference statistics from JSON."""
    from driftveil.core.pact import DriftVeil
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pact = DriftVeil(reference_df)
    
    for item in data.get("contracts", []):
        target = item["target"]
        # Convert list back to tuple for pairs
        if item["target_type"] == "pair" and isinstance(target, list):
            target = tuple(target)
            
        pact.contracts.append({
            "target_type": item["target_type"],
            "target": target,
            "contract_type": item["contract_type"],
            "params": item["params"],
            "severity": item["severity"],
            "ref_stats": item["ref_stats"]
        })
        
    return pact
