import pandas as pd
import numpy as np

def generate_dataset(file_name="data/synthetic_cost_dataset.csv", n_samples=1000, random_state = 42):
    """
    Generates a synthetic dataset for deep learning project cost estimation with highly randomized and advanced mathematical relationships.
    
    Parameters:
        file_name (str): Name of the CSV file to save the dataset.
        n_samples (int): Number of samples to generate.
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Constants for cost calculations
    RESOURCE_COST_AWS = 0.1  # $ per compute hour for AWS
    RESOURCE_COST_AZURE = 0.12  # $ per compute hour for Azure
    DATA_COST_PER_GB = 0.02  # $ per GB for data
    LABOR_COST_PER_HOUR = 50  # $ per hour for labor
    STORAGE_COST_PER_GB = 0.023  # $ per GB for data storage
    CARBON_COST_PER_KWH = 0.05  # $ per kWh

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
        "dataset_complexity": np.random.gamma(2, 0.5, n_samples),  # Gamma distribution for complexity
        "num_classes": np.random.randint(2, 100, n_samples),
        "augmentation": np.random.choice([0, 1], n_samples),  # 0 = No, 1 = Yes
        "data_format": np.random.choice(["Images", "Text", "Structured"], n_samples),
        
        # Infrastructure-related variables
        "compute_hours": np.random.poisson(100, n_samples),  # Poisson distribution for compute hours
        "cloud_provider": np.random.choice(["AWS", "Azure"], n_samples),
        "hardware_type": np.random.choice(["NVIDIA A100", "Tesla V100", "TPU v3"], n_samples),
        "energy_consumption_kwh": np.random.lognormal(2, 0.5, n_samples),  # Log-normal distribution for energy usage
        
        # Project-related variables
        "project_scale": np.random.choice(["Small", "Medium", "Large"], n_samples),
        "time_constraints_days": np.random.exponential(90, n_samples),  # Exponential distribution for time constraints
        "domain_type": np.random.choice(["Healthcare", "Finance", "Retail", "Autonomous Vehicles"], n_samples),
        "regulatory_requirements": np.random.choice([0, 1], n_samples),  # 0 = No, 1 = Yes
        
        # External costs
        "labor_hours": np.random.beta(2, 5, n_samples) * 500,  # Beta distribution scaled to 500
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

    # Add derived variables with highly randomized relationships
    df["resource_cost_per_hour"] = df["cloud_provider"].apply(
        lambda x: RESOURCE_COST_AWS if x == "AWS" else RESOURCE_COST_AZURE
    )
    df["training_cost"] = (
        df["compute_hours"]
        * df["resource_cost_per_hour"]
        * np.sin(df["training_epochs"] / 10) ** 2  # Sinusoidal relationship with epochs
    )
    df["data_cost"] = (
        np.sqrt(df["dataset_size"]) * DATA_COST_PER_GB * np.random.uniform(0.8, 1.2, n_samples)  # Adding random noise
    )
    df["labor_cost"] = (
        (df["labor_hours"] ** 1.2) * LABOR_COST_PER_HOUR * (1 + np.cos(df["dataset_complexity"]) / 10)  # Complex cosine factor
    )
    df["storage_cost"] = (
        np.exp(df["data_storage_gb"] / 800) * STORAGE_COST_PER_GB * np.random.uniform(0.9, 1.1, n_samples)
    )
    df["carbon_cost"] = (
        np.log1p(df["energy_consumption_kwh"]) * CARBON_COST_PER_KWH * (1 + np.random.exponential(0.1, n_samples))  # Adding exponential randomness
    )

    # Regulatory adjustment
    df["regulatory_cost_adjustment"] = df["regulatory_requirements"].apply(lambda x: 1.2 if x == 1 else 1.0)

    # Total cost calculation
    df["total_cost"] = (
        (df["training_cost"] + df["data_cost"] + df["labor_cost"] + df["storage_cost"] + df["carbon_cost"])
        * df["regulatory_cost_adjustment"]
    ) * np.random.uniform(0.9, 1.1, n_samples)  # Adding randomness to total cost

    # Save to CSV
    df.to_csv(file_name, index=False)
    print(f"Dataset saved to {file_name}")

    return df

def load_synthetic_dataset(file_name="data/synthetic_cost_dataset.csv"):
    """
    Loads the synthetic dataset from a CSV file.
    
    Parameters:
        file_name (str): Name of the CSV file to load the dataset from.
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    df = pd.read_csv(file_name)
    print(f"Dataset loaded from {file_name}")
    return df

if __name__ == "__main__":
    # Generate and preview the dataset
    dataset = generate_dataset(n_samples=int(1e5))
    print(dataset.head())
