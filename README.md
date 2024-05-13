### Reinforcement Learning Project on Recommendation systems. 

#### Project Objective - 
Implement existing DRL Actor critic framework (in Research papers folder) on Movielens 1 million dataset
Paper implementation trained for 5000 epochs 
Our implementation trained for 1500 epochs. (model weights and outputs attached in MidSemester Submission folder)

1. Compare implementation of framework with non-RL recommendation methods
2. Transfer DRL framework to a different domain (implemented on Book recommendation system)

## Comparisons for MovieLens 1 million dataset with 
- Graph Convolutional networks
- KNN (K nearest neighbours)
- SLIM
- Collaborative filtering

all precision and NDGC scores calculated in individual .ipynb files uploaded in 'Comparisons' folder

## Transfer DRL to Book-Crossing Dataset 
- performed EDA on Book-crossing dataset 
- carried out data preprocessing (scaling columns, creating mappings, handling NaNs, type conversions)

main file is Books_DQRL.ipyb with all data EDA, preprocessing, loading, calling train.py and evalutations displayed. 

had to make changes to train.py and subsequent files in original github (movielens) in order to train model and create embeddings.
as the zipped folder of modified reporitory for too large to upload on github (125 MB with max limit for a file being 100 MB) 
I have only uploaded necessary data to understand the implementation and major files with modifications. 

