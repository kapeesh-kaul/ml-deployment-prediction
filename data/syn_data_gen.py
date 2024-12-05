import pandas as pd
import numpy as np
import random

def generate_synthetic_cost_dataset(file_name, n_samples):
    """
    Generates a synthetic dataset for deep learning project cost estimation and saves it to a CSV file.
    
    Parameters:
        file_name (str): Name of the CSV file to save the dataset.
        n_samples (int): Number of samples to generate.
    """
    # Constants for cost calculations
    RESOURCE_COST_AWS = 0.1  # $ per compute hour for AWS
    RESOURCE_COST_AZURE = 0.12  # $ per compute hour for Azure
    DATA_COST_PER_GB = 0.02  # $ per GB for data
    LABOR_COST_PER_HOUR = 50  # $ per hour for labor
    STORAGE_COST_PER_GB = 0.023  # $ per GB for data storage
    CARBON_COST_PER_KWH = 0.05  # $ per kWh
    AUGMENTATION_COST_MULTIPLIER = 1.2  # Multiplier for projects with augmentation

    # Generate synthetic data
    data = {
        # Model-related variables
        "model_type": np.random.choice(["CNN", "RNN", "Transformer", "GNN"], n_samples),
        "model_size": np.random.randint(1e6, 1e9, n_samples),  # Number of parameters
        "training_epochs": np.random.randint(5, 100, n_samples),
        "training_batch_size": np.random.choice([16, 32, 64, 128], n_samples),
        "pretrained": np.random.choice([0, 1], n_samples),  # 0 = No, 1 = Yes
        "num_layers": np.random.randint(3, 150, n_samples),  # Number of layers in the model
        "optimizer_type": np.random.choice(["SGD", "Adam", "RMSProp"], n_samples),
        
        # Dataset-related variables
        "dataset_size": np.random.randint(1e3, 1e6, n_samples),  # Number of samples
        "dataset_complexity": np.random.uniform(0.5, 1.5, n_samples),  # Entropy-like score
        "num_classes": np.random.randint(2, 100, n_samples),
        "augmentation": np.random.choice([0, 1], n_samples),  # 0 = No, 1 = Yes
        "data_format": np.random.choice(["Images", "Text", "Structured"], n_samples),
        
        # Infrastructure-related variables
        "compute_hours": np.random.randint(10, 1000, n_samples),
        "cloud_provider": np.random.choice(["AWS", "Azure"], n_samples),
        "hardware_type": np.random.choice(["NVIDIA A100", "Tesla V100", "TPU v3"], n_samples),
        "energy_consumption_kwh": np.random.uniform(50, 500, n_samples),  # kWh
        
        # Project-related variables
        "project_scale": np.random.choice(["Small", "Medium", "Large"], n_samples),
        "time_constraints_days": np.random.randint(30, 180, n_samples),
        "domain_type": np.random.choice(["Healthcare", "Finance", "Retail", "Autonomous Vehicles"], n_samples),
        "regulatory_requirements": np.random.choice([0, 1], n_samples),  # 0 = No, 1 = Yes
        
        # External costs
        "labor_hours": np.random.randint(50, 500, n_samples),  # Hours of labor
        "data_storage_gb": np.random.randint(10, 1000, n_samples),  # GB
        
        # Environmental costs
        "carbon_footprint_tons": np.random.uniform(0.5, 10, n_samples),  # Estimated CO2 emissions in tons
        "renewable_energy_usage": np.random.uniform(0.2, 1.0, n_samples),  # Fraction of energy from renewable sources
        
        # Outcome metrics
        "target_accuracy": np.random.uniform(0.7, 0.99, n_samples),  # Accuracy between 70% and 99%
        "inference_time_ms": np.random.uniform(5, 500, n_samples),  # Inference time in milliseconds
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add derived variables
    df["resource_cost_per_hour"] = df["cloud_provider"].apply(
        lambda x: RESOURCE_COST_AWS if x == "AWS" else RESOURCE_COST_AZURE
    )
    df["training_cost"] = df["compute_hours"] * df["resource_cost_per_hour"]
    df["data_cost"] = df["dataset_size"] * DATA_COST_PER_GB
    df["labor_cost"] = df["labor_hours"] * LABOR_COST_PER_HOUR
    df["storage_cost"] = df["data_storage_gb"] * STORAGE_COST_PER_GB
    df["carbon_cost"] = df["energy_consumption_kwh"] * CARBON_COST_PER_KWH

    # Augmentation cost adjustment
    df["training_cost"] *= df["augmentation"].apply(lambda x: AUGMENTATION_COST_MULTIPLIER if x == 1 else 1)

    # Regulatory adjustment (adds extra 20% cost for regulated projects)
    df["regulatory_cost_adjustment"] = df["regulatory_requirements"].apply(lambda x: 1.2 if x == 1 else 1.0)
    df["total_cost"] = (
        (df["training_cost"] + df["data_cost"] + df["labor_cost"] + df["storage_cost"] + df["carbon_cost"])
        * df["regulatory_cost_adjustment"]
    )

    # Add randomness to relationships
    df["total_cost"] *= np.random.uniform(0.9, 1.1, n_samples)  # Add +/- 10% noise

    # Save to CSV
    df.to_csv(file_name, index=False)
    print(f"Dataset saved to {file_name}")

    return df

if __name__ == "__main__":
    # Generate and preview the dataset
    dataset = generate_synthetic_cost_dataset(file_name="data/synthetic_cost_dataset.csv", n_samples=int(1e5))
    print(dataset.head())
