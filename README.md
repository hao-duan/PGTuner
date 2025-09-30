# PGTuner: An Efficient Framework for Automatic and Transferable Configuration Tuning of Proximity Graphs


---

## Project Structure

| Folder/File | Description                                                                                                                      |
|---|----------------------------------------------------------------------------------------------------------------------------------|
| **Data** | Stores the original vector datasets, the collected query performance data, and the data generated during runtime.                |
| **parameter_configuration_recommend** | Contains the implementation of of the PCR model and the data generated during runtime.                                           |
| **query_performance_predict** | Contains the implementation of of the QPP model and model transfer, and the data generated during runtime.                       |
| **utils** | Contains the baisc utility codes, such as data reading/writing and data sampling.                                                |
| **hnswlib** | The library of the HNSW index .                                                                                                  |
| **NSG_KNNG** | Contains the the implementation of the NSG index, the PGTuner‑related codes and the data generated during runtime.               |
| **Other files in the project root** | Some core functionalities: brute‑force nearest neighbors search, query performance collection, dataset features extraction, etc. |

> **Note:** PGTuner is mainly implemented for the **HNSW** index. For the **NSG** index, a small number of specific implementation files are provided, which typically contain `nsg` in their filenames.
---

## Usage Instructions

###  Environment Setup

- Create an environment with **Python=3.9**, then install dependencies:

```bash
pip install --extra-index-url https://pypi.nvidia.com cuml-cu11
pip install cupy-cuda11x
pip install -r requirements.txt
```

> **Note:** The reference environment uses **CUDA 11.6** and **PyTorch 1.13**. Please choose compatible versions of **cuML**, **CuPy**, and **PyTorch** for your setup.  
To install **cuML** and **CuPy**:
- **For CUDA 11.x**: use the commands shown above.  
- **For CUDA 12.x**: you can use the command shown below.
```bash
pip install --extra-index-url https://pypi.nvidia.com cuml-cu12==25.8.*
```
> You can also refer to: <https://docs.rapids.ai/install/#prerequisites>, <https://pypi.org/search/?q=cupy&page=1>.

---

###  Data Preparation

Create three subdirectories under `/PGTuner/Data`: `Base`, `Query`, `GroundTruth`.

```bash
mkdir -p ./Data/Base
mkdir -p ./Data/Query
mkdir -p ./Data/GroundTruth
```

For each dataset you will use, further create subdirectories under `Data/Base`, `Data/Query`, and `Data/GroundTruth` named after the dataset, which store the base vectors, query vectors, and ground-truth nearest neighbors of the dataset, respectively. For example:

```bash
mkdir -p ./Data/Base/tiny
mkdir -p ./Data/Query/tiny
mkdir -p ./Data/GroundTruth/tiny
```

Then **Rename files** as follows:
- **Query vectors**: `dim.fvecs` (or `.bvecs`).
- **Base vectors** and **Ground Truth**: `level_num_dim.fvecs` (or `.bvecs`) and `level_num_dim.ivecs` respectively.

Where:
- $\mathrm{size}$ is the number of base vectors;
- $\mathrm{level} = \lfloor \log_{10}(\frac{\mathrm{size}}{100000}) \rfloor$;
- $\mathrm{num} = \frac{\mathrm{size}}{100000 \cdot 10^{\mathrm{level}}}$;
- dim is the vector dimension.

**Example:** For dataset `tiny1M` with `1M` base vectors and dimension `384`:
- Base vectors file: `1_1_384.fvecs`
- GroundTruth file: `1_1_384.ivecs`
- Query vectors file: `384.fvecs`

---

###  Query Performance Collection

From the project root `PGTuner`:

```bash
python query_performance_collect.py
```

For the **NSG index** (inside `PGTuner/NSG_KNNG`):

```bash
cd ./NSG_KNNG
python nsg_query_performance_collect.py
```

> **Note:** Update the target dataset name in the file  before running.

---

### Dataset Feature Extraction

```bash
python get_LID_feature.py
python get_DS_feature.py
python get_DR_feature.py
```

> **Note:** Update the target dataset name in each file before running.

---

### QPP Model Training and Transfer

Enter the directory:

```bash
cd ./query_performance_predict
```

**Data generation and preparation:**

```bash
python data_process.py          
python data_normalized.py
```
For the **NSG index**, run:
```bash
python data_process_nsg.py         
python data_normalized_nsg.py
```
The subsequent steps are similar.

**Train the QPP model:**

```bash
python train.py
```

**Transfer the QPP model** to a new dataset `dataset_name`:

```bash
python active_learning.py --dataset-name dataset_name --experiment-mode main
```

See `query_performance_predict/Args.py` for `experiment-mode` options.

**Successive transfer across multiple datasets** (`dataset_name1 → dataset_name2 → dataset_name3 → ...`):

```bash
python active_learning_successive.py --dataset-name dataset_name1 --experiment-mode dataset_change
python active_learning_successive.py --dataset-name dataset_name2 --lats-dataset-name dataset_name1 --experiment-mode dataset_change
python active_learning_successive.py --dataset-name dataset_name3 --lats-dataset-name dataset_name2 --experiment-mode dataset_change
# ...
```

---

### PCR Model Training and Online Tuning

Enter the directory:

```bash
cd ./parameter_configuration_recommend
```
---

**Train the PCR model:**
```bash
python train.py
```

**Online Tuning and Evaluation:**

Default target recalls: `[0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]`.

```bash
python evaluate.py --dataset-name dataset_name --experiment-mode main
```

**Generate recommended parameter configurations:**

```bash
python generate_recommended_configurations.py --dataset-name dataset_name --experiment-mode main
```

**Plot training/evaluation curves:**

```bash
python draw_training_results.py --training-mode train
# or
python draw_training_results.py --dataset-name dataset_name --training-mode evaluate --experiment-mode main
```

### Verify the query performance of recommended configurations

From the project root `PGTuner`:

```bash
python query_performance_verify.py --dataset-name dataset_name --experiment-mode main
```

For the **NSG** index:

```bash
cd ./NSG_KNNG
python nsg_query_performance_verify.py --dataset-name dataset_name --experiment-mode main
```

---

## Examples

There are some data provided under `PGTuner/Data`:
- **Dataset features**: `data_feature.csv`
- **Query performance data (5 base datasets)**: `index_performance_train.csv`
- **Query performance data (6 new datasets)**: `index_performance_test_main.csv`

Additionally, the **pretrained PCR model** is available in `PGTuner/parameter_configuration_recommend` for out‑of‑the‑box online tuning.

**Quick start** (from the project root  `PGTuner`):

```bash
# Run once for setup
chmod +x example.sh
cd ./query_performance_predict
python data_process.py
python data_normalized.py
python train.py
cd ..

# pipeline run
./example.sh tiny main
```

---

## Version Notes

In the current version:
- `active_learning.py`, `active_learning_nsg.py`, and `active_learning_successive.py` obtain the query performance data of the selected unlabeled configurations **from pre‑generated data files** (produced by prior GridSearch runs).
- In `evaluate.py` and `evaluate_nsg.py`, the query performance corresponding to the configuration `(20, 4, 10)` is also read **from existing data**.

> This way does not affect the running logic of PGTuner. The future version will obtain the query performance of these configurations by **constructing indexes on the fly**, which better suitable for real-world applications.
