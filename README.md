# GCNCMI

基于图神经网络的CircRNA-miRNA相互作用预测

# Requerments

numpy==1.16.6


pandas==1.1.5


scikit-learn==0.24.2

tensorflow-gpu==1.15.0


# Introduction


准确预测circRNA和miRNA之间的相互作用对于研究基因表达和基因对疾病的调控具有重要作用。然而，大多数 circRNA 和 miRNA 之间的相互作用仍不清楚。通过常规生物学实验挖掘circRNA-miRNA相互作用既费时又费力。因此，迫切需要一种计算方法来挖掘circRNA和miRNA之间的相互作用。在本文中，我们提出了一种图神经网络方法来预测 circRNAs 和 miRNAs (GCNCMI) 之间的潜在相互作用。我们使用图神经网络来挖掘circRNA和miRNA之间的高阶交互。GCNCMI首先挖掘图卷积神经网络中相邻节点的潜在交互，然后在图卷积层上递归传播交互信息，最后连接嵌入的表示由每一层生成以进行最终预测。在五折交叉验证中，GCNCMI 的 AUC 最高为 0.9312，AUPR 最高为 0.9412。此外，对hsa-miR-622和hsa-miR-149-3p这两种miRNA的案例分析表明，我们的模型对预测circRNA和miRNA之间的相互作用有很好的效果。 

<!-- #Citation
If you want to use our codes and datasets in your research, please cite:


CircRNA-miRNA interaction Prediction Based on
Graph Neural Network -->
# Dataset

Our dataset contains 821 miRNAs and 2115 circRNAs, which we divide into a 5-fold cross-validation dataset.


