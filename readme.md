# Multi-Input Deep Learning Graph Model to Classify for Alzheimer’s Disease

## University of London BSc Computer Science CM3070 Final Project

### Pragya Modi
### 190308090

----

## Project Overview

This project aims to build a neural network based graph model for Alzheimer's classification. While there are various studies already aiming to solve the problem, very few take into account the real world scenario. Most studies focus on building an image classification model using MRI scans. However, in the real world, patient information like family history, disease history, and tests are taken into account alongside MRI scans.

Therefore, this project aims to build a graph model taking into account both patient information (features) and MRI scans (images).

## Dataset Used

The dataset used for this project is the OASIS brain scan dataset `[1]`. The dataset allowed downloading images data in NIFTI format and all the features' data was available across multiple CSV files.

*NIFTI Format:*

NIFTI is an open-source file format commonly used for storing brain imaging data obtained from an MRI scan.


## Replicating The Repo

The dataset used for this project is from the OASIS-3 dataset `[1]`. The first notebook uses the main dataset from this dataset with is of 19GB and creates a 50MB TensorFlow dataset. This TensorFlow Dataset is used ahead in all other notebooks. Hence, all but the first notebooks are replicable and can be run from downloading the repository. However, the first notebook is not replicable because of the original dataset size.


## Results

This project built a multi-input graph model using MRI scans and patient demographics and Mini Mental State Examination (MMSE) results. This model was then compared to single input model with just the MRI scans.

| Model          | Validation Accuracy |
|----------------|---------------------|
| Graph Model    | 95.51%              |
| Image Model    | 91.33%              |
| Features Model | 72.51%              |

## References

[1] Pamela J. LaMontagne, Tammie LS. Benzinger, John C. Morris, Sarah Keefe, Russ Hornbeck, Chengjie Xiong, Elizabeth Grant, Jason Hassenstab, Krista Moulder, Andrei G. Vlassenko, Marcus E. Raichle, Carlos Cruchaga, and Daniel Marcus. 2019. OASIS-3: Longitudinal Neuroimaging, Clinical, and Cognitive Dataset for Normal Aging and Alzheimer Disease. Radiology and Imaging. DOI:https://doi.org/10.1101/2019.12.13.19014902
