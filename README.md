This repository contains the code to replicate the experiments described in the paper "A Deep Survival EWAS approach estimating risk profile based on pre-diagnostic DNA methylation: an application to Breast Cancer Time to Diagnosis", PLOS Computational Biology (2022), authored by Massi M.C., Dominoni L., Ieva F and Fiorito G.

In particular:
Preprocessing.R : R script to create the preprocessed datasets from the raw data given, selecting the EZH2 sites subset and aggregating the sites in islands. 


Input Feature Agglomeration and Output Results construction.ipynb : Python Google Colaboratory notebook composed of two parts: 
PART I contains the code to apply CpG Islands agglomeration into Features, as described in Section “Feature agglomeration” in Materials and Methods. This part contains the code to generate S4 Fig., i.e. the plot of the granularity of the clusters
PART II should be run after Survival_model.ipynb (see below). It contains the code to  plot the weights profiles of each island for both Survival EWAS (Fig.2 in the paper) and the Standard approach (S1 Fig.). Finally, it  extracts the islands of the ranked first feature. 
Both parts exploit functions in “feature_clustering_func.py” custom python module.
feature_clustering_func.py: Python custom module that contains the helper functions to run “Feature_clustering.ipynb”. It has functions to transform a dataset according to a given feature clustering (transform_clustering), create the dataset of the results (construct_dataset_results), plot the importance of each island according to the deep survival approach (plot_results), plot the importance of each island according to the standard approach (plot_results_cox), plot the histogram of the numerosity of the clusters for each clustering dimension (plot_granularity).


Survival_model.ipynb : Python Google Colaboratory notebook that contains the code to run all the survival modeling analyses described in the paper. 
In particular, in the first part of this notebook (“Deep Survival Modeling”) we perform the analyses described in Deep Survival Model Architecture and Training (including model pretraining) and Algorithm Design and Optimization section in Materials and Methods, i.e. model building and training of the tested Deep survival Neural Networks (NN), with the subsequent application of SHAP to estimate weights profile, and it computes the performance (cf. Performance Measures in Materials and Methods) for the tested Deep Survival NN. This notebook also contains the code to generate Fig.2 (panel A) and Fig. 3 (panel A).
In the second part “Multivariate Cox Model”, we perform the benchmark comparison with multivariate cox model for all input features granularity. 
The notebook exploits functions in the custom python module “survival_model_func.py”.
survival_model_func.py: Python script that contains the helper functions to run “Survival_model.ipynb”.
It contains functions to compute the c-index (c_statistic_harrell), define the deepsurv model (adapted from the notebook at https: //nbviewer.org/github/sebp/survival-cnn-estimator/blob/master/tutorial_tf2.ipynb), pretrain the model (pretrain_one_layer and pretraining), train the model (surv_training), apply the Shap algorithm for each background dataset (explanation_algorithm), compute the KT-stability (normalized_kendall_tau_distance and KT_stab), calculate the mean importance values and ranking (mean_ranking).


kaplan_curves.R : R script to compute log-rank p-values and hazard ratios statistics and to plot the Kaplan Meier curves for each island belonging to the ranked first feature.
Plot_performances.R : R script to create forest plots of the C-index and KT-stability performances across all models and dimensionalities.
 Standard_model.R : R script to train and measure the performance of the Cox models using one island at a time. Its results are used for the application of the WKS for the standard EWAS approach.
Enrichment_analysis.R : R script to perform the WKS tests for both the standard and deep survival approaches using the importance values as weights.

Versioning and libraries: 
Python source code was implemented with Python 3.7.12. Main dependencies include Scikit-learn (version 0.24.1 and 1.0.1), Scikit-survival (version 0.15.0.post0), TensorFlow (version 2.8.0), PySurvival (version 0.1.2), Shap (version 0.40.0), Keras, Pandas, Numpy, Matplotlib, SciPy, Statsmodels, Seaborn. 
R scripts are implemented in R version 4.0.5. Main packages to run the code are survival, ggplot2, KEGGREST, wks, IlluminaHumanMethylation450kanno.ilmn12.hg19.
