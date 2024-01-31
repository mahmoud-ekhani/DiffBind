# DiffBind
Python implementation of a denoising diffusion probabilistic model (DDPM) for predicting the binding affinity of a protein-ligand complex.

## Setting Up the Environment

To run the code in this repository, you'll need to create a Conda environment with specific dependencies. Below are the steps to set up the environment:

### Creating a Conda Environment

1. **Create a New Conda Environment:**  
   This command creates a new Conda environment named `diff_bind`.
   ```bash
   conda create -n diff_bind
   ```

2. **Activate the Environment:**  
   Once the environment is created, activate it using:
   ```bash
   conda activate diff_bind
   ```

### Installing Dependencies

Install the necessary dependencies by running the following commands in your activated environment:

- **PyTorch with CUDA Toolkit:**
  ```bash
  conda install pytorch cudatoolkit=10.2 -c pytorch
  ```

- **PyTorch Lightning:**
  ```bash
  conda install -c conda-forge pytorch-lightning
  ```

- **Weights & Biases (wandb):**
  ```bash
  conda install -c conda-forge wandb
  ```

- **RDKit:**
  ```bash
  conda install -c conda-forge rdkit
  ```

- **Biopython:**
  ```bash
  conda install -c conda-forge biopython
  ```

- **ImageIO:**
  ```bash
  conda install -c conda-forge imageio
  ```

- **SciPy:**
  ```bash
  conda install -c anaconda scipy
  ```

- **PyTorch-Scatter:**
  ```bash
  conda install -c pyg pytorch-scatter
  ```

- **OpenBabel:**
  ```bash
  conda install -c conda-forge openbabel
  ```

### Final Steps

After installing all dependencies, your environment is ready to run the code in this repository.

## Binding MOAD
This section describes how to prepare the Binding MOAD dataset for processing.
### Data preparation
1. **Download the Dataset:**
    Run the following commands in your terminal to download the necessary files:
    ```bash
    wget http://www.bindingmoad.org/files/biou/every_part_a.zip
    wget http://www.bindingmoad.org/files/biou/every_part_b.zip
    wget http://www.bindingmoad.org/files/csv/every.csv

    unzip every_part_a.zip
    unzip every_part_b.zip
    ```
2. **Process the raw data:**
    Use the provided Python script to process the raw data. Replace '<bindingmoad_dir>'
    with the path to the directory where you've downloaded the Binding MOAD files:
    ``` bash
    python -W ignore process_bindingmoad.py <bindingmoad_dir>
    ```
    - *Optional:* To create a dataset with only C-alpha (Cα) pocket representation, add the 
    '--ca_only' flag to the command:
        ```bash
        python -W ignore process_bindingmoad.py <bindingmoad_dir> --ca_only
        ```
        This flag will configure the dataset to focus on the C-alpha atoms in the protein structure.