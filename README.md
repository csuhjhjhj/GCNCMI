# GCNCMI
CircRNA-miRNA interaction Prediction Based on Graph Neural Network

# Requerments

numpy==1.16.6


pandas==1.1.5


scikit-learn==0.24.2

tensorflow-gpu==1.15.0


# Introduction

Accurate prediction of the interaction  between circRNA and miRNA plays an essential role in studying gene expression and the regulation of genes on diseases. However, the interaction  between most circRNAs and miRNAs are still unclear. It is time-consuming and labor-intensive to mine circRNA-miRNA interaction  by conventional biological experiments. Therefore, a computational method is urgently needed to mine the interaction between circRNAs and miRNAs. In this paper,we propose a graph neural network method to predict the potential interaction  between circRNAs and miRNAs (GCNCMI). We use graph neural networks to mine higher-order interaction  between circRNAs and miRNAs.GCNCMI first mines the potential interactions of adjacent nodes in the graph convolutional neural network,and then recursively propagates interaction  information on the graph convolutional layers, finally it connects the embedded representations generated by each layer to make the final prediction. In the five-fold cross-validation, GCNCMI has the highest AUC of 0.9312 and the highest AUPR of 0.9412. In addition, the case analysis of two miRNAs, hsa-miR-622 and hsa-miR-149-3p, showed that our model has a good effect on predicting the interaction  between circRNA and miRNA.


<!-- #Citation
If you want to use our codes and datasets in your research, please cite:


CircRNA-miRNA interaction Prediction Based on
Graph Neural Network -->
# Dataset

Our dataset contains 821 miRNAs and 2115 circRNAs, which we divide into a 5-fold cross-validation dataset.


