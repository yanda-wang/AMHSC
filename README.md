# AMHSC

# 概述

该项目是论文《一种自适应记忆神经网络多跳读取与覆盖度机制结合的药物推荐模型》所提出的AMHSC模型的代码实现。

AMHSC模型以病人对应的电子医疗记录（Electronic Healthcare Records, EHRs）为依据生成药物处方，为医生提供临床决策支持。模型首先对EHRs中的时序模型进行编码，并基于编码结果构建记忆神经网络（Memory Neural Network, MemNN）。然后，模型会通过对MemNN进行基于注意力机制的多跳读取，获取其中蕴含的上下文信息，用于完善病人表示向量，作为药物推荐的依据。在此过程中，模型利用覆盖度机制进行数据迭代读取过程中的MemNN数据过滤以及注意力权重调整，保证注意力权重适当分散于MemNN中的重要数据范围内。此外，模型依据病人自身数据，自适应决定MemNN读取次数，避免过度读取或是数据读取不充分对药物推荐性能的影响。

# Overview

This is the implement for the model AMHSC presented in our paper "Adaptive Multi-Hop Reading on Memory Neural Network with Selective Coverage Mechanism for Medication Recommendation" (in Chinese).

AMHSC makes prescirptions for patients based on their Electronic Healthcare Records (EHRs), and assists doctors in clinical decision-making. The model firstly encodes temporal patterns contaiend in EHRs, and builds key-value Memory Neural Network (MemNN) to the encoding results. Then AMHSC conducts attention-based multi-hop reading on MemNN to read contextual information, which is used to refine patient representations for the recommendation. During this process, AMHSC uses selective coverage mechanism to filter data in MemNN and adjusts the attention weights, making sure the attention is appropriately adjusted over the selected important data. Moreover, AMHSC adaptively decides the number of hops of reading on MemNN for each patient based on their EHRs, which helps to avoid the negative effectes related to over-reading or under-reading.

# 代码实现要求

Pytorch 1.1

Python 3.8

# Requirement

Pytorch 1.1

Python 3.8

# 数据

模型使用临床医疗数据集[MIMIC-III](https://mimic.physionet.org)开展相关实验，用以验证模型性能。MIMIC-III是真实的临床EHRs数据，收集了超过四万名病人在住院期间的临床信息。AMHSC模型使用其中的诊断结果（Diagnoses）以及手术治疗方法（Procedures）作为输入，预测病人所对应的药物集合。

模型所需数据的生成与整理过程如下：

首先，从MIMIC-III官网下载如下三份数据：

DIAGNOSES_ICD.csv

PROCEDURES_ICD.csv

PRESCRIPTIONS.csv

然后，准备如下数据（已经上传至AMHSC/data）

drug-atc.csv

ndc2atc_level4.csv

ndc2rxnorm_mapping.txt

将上述所有文件放置到工程下的data文件中，运行代码获取数据：

python DataProcessing

得到的文件包括：

patient_records.pkl：以医疗代码形式存储的病人EHRs

voc_final.pkl：医疗代码的字典

records_final.pkl：字典化后的病人EHRs

ehr_adj_final.pkl：表明两种药物是否被同时开具给同一个病人的矩阵

# Data

All experiments are conducted based on [MIMIC-III](https://mimic.physionet.org), which is a real-world clinical dataset that collectes EHRs related to over 40,000 patients. AMHSC uses Diagnoses and Procedures as the input to predict ground truth medications.

To prepare all datasets requried by AMHSC, firstly downloads the following three files from MIMIC-III:

DIAGNOSES_ICD.csv

PROCEDURES_ICD.csv

PRESCRIPTIONS.csv

Then prepare the following datasets, which you can find in AMHSC/data:

drug-atc.csv

ndc2atc_level4.csv

ndc2rxnorm_mapping.txt

Put all these files in a file naemd as 'data' under the project, and run:

python DataProcessing

You can find the following files:

patient_records.pkl：EHRs in the form of medical codes

voc_final.pkl：vocalbularies of medical codes

records_final.pkl：EHRs after the vocalbularization

ehr_adj_final.pkl：matrix that indicates co-occurrence of medications


# 运行模型

(1) 首先进行基于固定MemNN读取次数的药物推荐，训练得到的模型会被保存到参数--save_model_dir指定的目录：

python run_model --mode fixed_hop_train --save_model_dir data/model/fixed_hop_model 

(2) 然后，基于上述训练结果，生成病人对应的最佳MemNN读取次数，作为后续自适应药物推荐训练时的监督信息。这里使用参数--load_model_name指定所加载的模型，得到的数据会被保存到参数--save_supervised_hop_dir指定的目录：

python run_model --mode generate_adaptive_hop --load_model_name data/fixed_hop_model/your_model --save_supervised_hop_dir data/supervised_hop_data

(3) 接着，基于得到的数据进行自适应MemNN读取模型的训练。这里使用--patient_records_file_supervised_hop指定所使用的监督信息，训练得到的模型会被存储到参数--save_model_dir指定的目录：

python run_model --mode adaptive_hop_train --load_model_name data/fixed_hop_model/your_model --patient_records_file_supervised_hop data/supervised_hop_data/records_supervised_hop.pkl --save_model_dir data/model/adaptive_model

(4) 最后，加载模型以对其性能进行评估，这里使用参数--load_model_name指定所加载的模型

python run_model.py --mode eval --patient_records_file_supervised_hop data/supervised_hop_data/records_supervised_hop.pkl --load_model_name data/model/adaptive_model/your_model --save_predict_results_dir data

# code

After preparing all the required datasets:

(1) 

