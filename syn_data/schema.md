# Highly Randomized Synthetic Cost Estimation Dataset

This document describes the columns in the synthetic dataset for estimating costs in deep learning projects. The dataset includes base variables, derived variables, and second-level derived variables, utilizing advanced mathematical relationships and non-normal distributions to ensure realistic interdependencies.

---

## Base Columns

### Model-Related Variables
- **`model_type`**: Categorical variable indicating the type of deep learning model:
  - Values: {CNN, RNN, Transformer, GNN}.
- **`model_size`**: Number of trainable parameters in the model:
  - \( 10^6 \leq \text{model\_size} \leq 10^9 \) (in parameters).
- **`training_epochs`**: Number of training iterations:
  - \( 5 \leq \text{training\_epochs} \leq 100 \).
- **`training_batch_size`**: Size of the mini-batches used during training:
  - Values: {16, 32, 64, 128}.
- **`pretrained`**: Boolean variable indicating whether a pre-trained model was used:
  - Values: {0 (No), 1 (Yes)}.
- **`num_layers`**: Number of layers in the model:
  - \( 3 \leq \text{num\_layers} \leq 150 \).
- **`optimizer_type`**: Categorical variable indicating the optimizer used:
  - Values: {SGD, Adam, RMSProp}.

---

### Dataset-Related Variables
- **`dataset_size`**: Number of samples in the dataset:
  - \( 10^3 \leq \text{dataset\_size} \leq 10^6 \).
- **`dataset_complexity`**: Complexity of the dataset, modeled with a gamma distribution:
  - \( \text{dataset\_complexity} \sim \text{Gamma}(\alpha=2, \beta=0.5) \).
- **`num_classes`**: Number of classes in the dataset:
  - \( 2 \leq \text{num\_classes} \leq 100 \).
- **`augmentation`**: Boolean variable indicating whether data augmentation was applied:
  - Values: {0 (No), 1 (Yes)}.
- **`data_format`**: Categorical variable indicating the format of the dataset:
  - Values: {Images, Text, Structured}.

---

### Infrastructure-Related Variables
- **`compute_hours`**: Total compute hours required for training, modeled with a Poisson distribution:
  - \( \text{compute\_hours} \sim \text{Poisson}(\lambda=100) \).
- **`cloud_provider`**: Categorical variable indicating the cloud platform used:
  - Values: {AWS, Azure}.
- **`hardware_type`**: Categorical variable indicating the type of hardware used:
  - Values: {NVIDIA A100, Tesla V100, TPU v3}.
- **`energy_consumption_kwh`**: Energy consumed during training, modeled with a log-normal distribution:
  - \( \text{energy\_consumption\_kwh} \sim \text{LogNormal}(\mu=2, \sigma=0.5) \).

---

### Project-Related Variables
- **`project_scale`**: Categorical variable indicating the scale of the project:
  - Values: {Small, Medium, Large}.
- **`time_constraints_days`**: Deadline for project completion, modeled with an exponential distribution:
  - \( \text{time\_constraints\_days} \sim \text{Exponential}(\lambda=1/90) \).
- **`domain_type`**: Categorical variable indicating the project domain:
  - Values: {Healthcare, Finance, Retail, Autonomous Vehicles}.
- **`regulatory_requirements`**: Boolean variable indicating if the project is in a regulated domain:
  - Values: {0 (No), 1 (Yes)}.

---

### External Costs
- **`labor_hours`**: Total labor hours required for the project, modeled with a beta distribution:
  - \( \text{labor\_hours} \sim \text{Beta}(\alpha=2, \beta=5) \times 500 \).
- **`data_storage_gb`**: Amount of data storage used (in GB):
  - \( 10 \leq \text{data\_storage\_gb} \leq 1000 \).

---

### Environmental Costs
- **`carbon_footprint_tons`**: Carbon footprint of the project:
  - \( 0.5 \leq \text{carbon\_footprint\_tons} \leq 10 \).
- **`renewable_energy_usage`**: Proportion of energy sourced from renewables:
  - \( 0.2 \leq \text{renewable\_energy\_usage} \leq 1.0 \).

---

### Outcome Metrics
- **`target_accuracy`**: Desired model accuracy:
  - \( 0.7 \leq \text{target\_accuracy} \leq 0.99 \).
- **`inference_time_ms`**: Time for a single inference (in milliseconds):
  - \( 5 \leq \text{inference\_time\_ms} \leq 500 \).

---

## Derived Columns

### Cost Components

1. **`resource_cost_per_hour`**:
   \[
   \text{resource\_cost\_per\_hour} =
   \begin{cases} 
   0.10 & \text{if cloud\_provider = "AWS"} \\
   0.12 & \text{if cloud\_provider = "Azure"} 
   \end{cases}
   \]

2. **`training_cost`**:
   \[
   \text{training\_cost} = \text{compute\_hours} \times \text{resource\_cost\_per\_hour} \times \sin^2\left(\frac{\text{training\_epochs}}{10}\right)
   \]

3. **`data_cost`**:
   \[
   \text{data\_cost} = \sqrt{\text{dataset\_size}} \cdot 0.02 \cdot U(0.8, 1.2)
   \]
   - \( U(0.8, 1.2) \): Uniform random noise.

4. **`labor_cost`**:
   \[
   \text{labor\_cost} = (\text{labor\_hours})^{1.2} \cdot 50 \cdot \left(1 + \frac{\cos(\text{dataset\_complexity})}{10}\right)
   \]

5. **`storage_cost`**:
   \[
   \text{storage\_cost} = e^{\text{data\_storage\_gb} / 800} \cdot 0.023 \cdot U(0.9, 1.1)
   \]

6. **`carbon_cost`**:
   \[
   \text{carbon\_cost} = \ln(1 + \text{energy\_consumption\_kwh}) \cdot 0.05 \cdot (1 + \text{Exp}(0.1))
   \]
   - \( \text{Exp}(0.1) \): Exponential random noise.

---

### Adjustments

1. **`regulatory_cost_adjustment`**:
   \[
   \text{regulatory\_cost\_adjustment} =
   \begin{cases} 
   1.2 & \text{if regulatory\_requirements = 1} \\
   1.0 & \text{otherwise}
   \end{cases}
   \]

---

### Total Cost

2. **`total_cost`**:
   \[
   \text{total\_cost} = 
   (\text{training\_cost} + \text{data\_cost} + \text{labor\_cost} + \text{storage\_cost} + \text{carbon\_cost}) 
   \times \text{regulatory\_cost\_adjustment} \cdot U(0.9, 1.1)
   \]
   - \( U(0.9, 1.1) \): Uniform random noise.

---

## Notes
- **Distributions:**
  - Gamma, Poisson, Log-normal, Exponential, and Beta distributions are used for various variables to mimic real-world variability.
- **Nonlinear Relationships:** Sinusoidal, cosine, exponential, and logarithmic functions are applied to add complexity.
- **Randomness:** Uniform and exponential random noise enhance realism.

---

## Output File
- The dataset is saved as `randomized_synthetic_cost_dataset.csv`.
