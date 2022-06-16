# GCNCMI

基于图神经网络的CircRNA-miRNA相互作用预测

# Requerments

numpy==1.16.6


pandas==1.1.5


scikit-learn==0.24.2

tensorflow-gpu==1.15.0


# Introduction
环状 RNA (circRNA) 和 microRNA (miRNA) 之间的相互作用已被证明可以改变基因表达并调节疾病基因。由于传统的实验方法既费时又费力，因此大多数 circRNA-miRNA 相互作用在很大程度上仍然未知。开发计算方法来大规模探索 circRNA 和 miRNA 之间的相互作用可以帮助弥合这一差距。在本文中，我们提出了一种名为 GCNCMI 的基于图卷积神经网络的方法来预测 circRNA 和 miRNA 之间的潜在相互作用。 GCNCMI首先挖掘图卷积神经网络中相邻节点的潜在交互，然后递归地在图卷积层上传播交互信息。最后，它结合每一层生成的嵌入表示来进行最终预测。在五折交叉验证中，GCNCMI 达到了 0.9312 的最高 AUC 和 0.9412 的最高 AUPR。此外，对hsa-miR-622和hsa-miR-149-5p这两种miRNA的案例研究表明，我们的模型对预测circRNA-miRNA相互作用有很好的效果。


<!-- #Citation
If you want to use our codes and datasets in your research, please cite:


CircRNA-miRNA interaction Prediction Based on
Graph Neural Network -->
# Dataset

Our dataset contains 821 miRNAs and 2115 circRNAs, which we divide into a 5-fold cross-validation dataset.

# Usage
config目录下的GCNCMI.conf文件可以修改模型的参数
运行GCNCMI中的main函数，开始训练。
