# Synthetic Cost Estimation Dataset

This document describes the columns in the synthetic dataset generated for estimating costs in deep learning projects. The dataset includes base variables, derived variables, and second-level derived variables with mathematical relationships and ranges.

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
- **`dataset_complexity`**: Complexity of the dataset (entropy-like score):
  - \( 0.5 \leq \text{dataset\_complexity} \leq 1.5 \).
- **`num_classes`**: Number of classes in the dataset:
  - \( 2 \leq \text{num\_classes} \leq 100 \).
- **`augmentation`**: Boolean variable indicating whether data augmentation was applied:
  - Values: {0 (No), 1 (Yes)}.
- **`data_format`**: Categorical variable indicating the format of the dataset:
  - Values: {Images, Text, Structured}.

---

### Infrastructure-Related Variables
- **`compute_hours`**: Total compute hours required for training:
  - \( 10 \leq \text{compute\_hours} \leq 1000 \).
- **`cloud_provider`**: Categorical variable indicating the cloud platform used:
  - Values: {AWS, Azure}.
- **`hardware_type`**: Categorical variable indicating the type of hardware used:
  - Values: {NVIDIA A100, Tesla V100, TPU v3}.
- **`energy_consumption_kwh`**: Energy consumed during training (in kilowatt-hours):
  - \( 50 \leq \text{energy\_consumption\_kwh} \leq 500 \).

---

### Project-Related Variables
- **`project_scale`**: Categorical variable indicating the scale of the project:
  - Values: {Small, Medium, Large}.
- **`time_constraints_days`**: Deadline for project completion (in days):
  - \( 30 \leq \text{time\_constraints\_days} \leq 180 \).
- **`domain_type`**: Categorical variable indicating the project domain:
  - Values: {Healthcare, Finance, Retail, Autonomous Vehicles}.
- **`regulatory_requirements`**: Boolean variable indicating if the project is in a regulated domain:
  - Values: {0 (No), 1 (Yes)}.

---

### External Costs
- **`labor_hours`**: Total labor hours required for the project:
  - \( 50 \leq \text{labor\_hours} \leq 500 \).
- **`data_storage_gb`**: Amount of data storage used (in GB):
  - \( 10 \leq \text{data\_storage\_gb} \leq 1000 \).

---

### Environmental Costs
- **`carbon_footprint_tons`**: Carbon footprint of the project (in tons):
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
   \text{training\_cost} = \text{compute\_hours} \times \text{resource\_cost\_per\_hour}
   \]
   - Adjusted by augmentation:
   \[
   \text{training\_cost} \times 1.2 \text{ if augmentation = 1}.
   \]

3. **`data_cost`**:
   \[
   \text{data\_cost} = \text{dataset\_size} \times 0.02 \, (\text{USD per GB}).
   \]

4. **`labor_cost`**:
   \[
   \text{labor\_cost} = \text{labor\_hours} \times 50 \, (\text{USD per hour}).
   \]

5. **`storage_cost`**:
   \[
   \text{storage\_cost} = \text{data\_storage\_gb} \times 0.023 \, (\text{USD per GB}).
   \]

6. **`carbon_cost`**:
   \[
   \text{carbon\_cost} = \text{energy\_consumption\_kwh} \times 0.05 \, (\text{USD per kWh}).
   \]

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
   \times \text{regulatory\_cost\_adjustment}
   \]
   - Random noise (\( \pm 10\% \)) is added to simulate variability.

---

## Notes
- The dataset captures logical interdependencies between variables.
- Derived columns build upon both base columns and other derived columns, ensuring realistic relationships.
- All variables are designed to mimic real-world scenarios in deep learning projects.

---

## Output File
- The dataset is saved as `data/synthetic_cost_dataset.csv`.
