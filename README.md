# DiffBind
Python implementation of a denoising diffusion probabilistic model (DDPM) for predicting the binding affinity of a protein-ligand complex.

### Binding MOAD
This section describes how to prepare the Binding MOAD dataset for processing.
#### Data preparation
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