"""Data loading utilities for onboarding lab assets."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def get_repo_root() -> Path:
    """Find the repository root by looking for markers."""
    current = Path(__file__).absolute()
    
    # Traverse up looking for markers
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "README.md").exists():
            return parent
    
    raise RuntimeError("Could not find repository root")


def get_assets_dir() -> Path:
    """Get the path to the assets directory."""
    repo_root = get_repo_root()
    assets_dir = repo_root / "Labs" / "Day_07_Advanced_Agent_Workflows" / "assets"
    
    if not assets_dir.exists():
        raise RuntimeError(f"Assets directory not found at {assets_dir}")
    
    return assets_dir


def load_json(filename: str) -> Any:
    """Load a JSON file from the assets directory with validation."""
    # Whitelist allowed files
    allowed_files = [
        "onboarding_docs.json",
        "roles_access_matrix.json", 
        "training_catalog.json",
        "new_hires_sample.json"
    ]
    
    if filename not in allowed_files:
        raise ValueError(f"File {filename} not in allowed list: {allowed_files}")
    
    assets_dir = get_assets_dir()
    file_path = assets_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Asset file not found: {file_path}")
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return data


def load_onboarding_docs() -> Dict[str, Any]:
    """Load and normalize onboarding documents."""
    data = load_json("onboarding_docs.json")
    
    # Ensure deterministic ordering
    if "policies" in data:
        data["policies"] = sorted(data["policies"], key=lambda x: x.get("id", ""))
    if "procedures" in data:
        data["procedures"] = sorted(data["procedures"], key=lambda x: x.get("id", ""))
    if "glossary" in data:
        data["glossary"] = dict(sorted(data["glossary"].items()))
    
    return data


def load_roles_access_matrix() -> Dict[str, Dict[str, Any]]:
    """Load and normalize role access matrix."""
    data = load_json("roles_access_matrix.json")
    
    # Normalize keys to lowercase for consistent lookups
    normalized = {}
    for role, access in data.items():
        role_lower = role.lower()
        normalized[role_lower] = access
        
        # Sort systems list for determinism
        if "systems" in access:
            access["systems"] = sorted(access["systems"])
        if "permissions" in access:
            access["permissions"] = sorted(access["permissions"])
    
    return normalized


def load_training_catalog() -> List[Dict[str, Any]]:
    """Load and normalize training catalog."""
    data = load_json("training_catalog.json")
    
    # Ensure it's a list
    if not isinstance(data, list):
        if "trainings" in data:
            data = data["trainings"]
        else:
            raise ValueError("Training catalog must be a list or contain 'trainings' key")
    
    # Sort by course_id for determinism
    data = sorted(data, key=lambda x: x.get("course_id", ""))
    
    # Normalize tags to lowercase and sort
    for training in data:
        if "tags" in training:
            training["tags"] = sorted([tag.lower() for tag in training["tags"]])
    
    return data


def load_new_hires() -> List[Dict[str, Any]]:
    """Load and normalize new hire sample data."""
    data = load_json("new_hires_sample.json")
    
    # Handle both list and dict formats
    if isinstance(data, dict) and "new_hires" in data:
        data = data["new_hires"]
    elif not isinstance(data, list):
        raise ValueError("New hires data must be a list or contain 'new_hires' key")
    
    # Sort by employee_id for determinism
    data = sorted(data, key=lambda x: x.get("employee_id", ""))
    
    # Normalize roles to lowercase
    for hire in data:
        if "role" in hire:
            hire["role"] = hire["role"].lower()
    
    return data


def validate_data() -> Dict[str, bool]:
    """Validate all data files can be loaded and have expected structure."""
    results = {}
    
    try:
        docs = load_onboarding_docs()
        results["onboarding_docs"] = (
            isinstance(docs, dict) and
            any(k in docs for k in ["policies", "procedures", "glossary"])
        )
    except Exception as e:
        results["onboarding_docs"] = False
        print(f"Error loading onboarding docs: {e}")
    
    try:
        matrix = load_roles_access_matrix()
        results["roles_access_matrix"] = (
            isinstance(matrix, dict) and
            len(matrix) > 0 and
            all(isinstance(v, dict) for v in matrix.values())
        )
    except Exception as e:
        results["roles_access_matrix"] = False
        print(f"Error loading roles access matrix: {e}")
    
    try:
        catalog = load_training_catalog()
        results["training_catalog"] = (
            isinstance(catalog, list) and
            len(catalog) > 0 and
            all(isinstance(t, dict) for t in catalog)
        )
    except Exception as e:
        results["training_catalog"] = False
        print(f"Error loading training catalog: {e}")
    
    try:
        hires = load_new_hires()
        results["new_hires"] = (
            isinstance(hires, list) and
            len(hires) > 0 and
            all(isinstance(h, dict) for h in hires)
        )
    except Exception as e:
        results["new_hires"] = False
        print(f"Error loading new hires: {e}")
    
    return results


if __name__ == "__main__":
    # Self-test
    print("Testing data loading utilities...")
    validation = validate_data()
    
    all_passed = all(validation.values())
    
    for name, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    if all_passed:
        print("\n✓ All data files loaded successfully!")
        
        # Print sample data
        docs = load_onboarding_docs()
        print(f"\nOnboarding docs keys: {list(docs.keys())}")
        
        matrix = load_roles_access_matrix()
        print(f"Roles available: {list(matrix.keys())}")
        
        catalog = load_training_catalog()
        print(f"Training courses: {len(catalog)}")
        
        hires = load_new_hires()
        print(f"Sample new hires: {len(hires)}")
    else:
        print("\n✗ Some data files failed to load")
        exit(1)