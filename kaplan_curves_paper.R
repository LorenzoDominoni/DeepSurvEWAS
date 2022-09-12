# packages
library(survival)
set.seed(1)

# load the dataset with the islands composing feature 120
dataset <- read.csv(file = 'best_features.csv', check.names = FALSE)
head(dataset)

# consider only cases
dataset = dataset[!is.na(dataset$time.to.disease),] 
dataset['event'] = rep(2,248) 

dataset=dataset[,-c(1,23)] # remove irrelevant columns
names=colnames(dataset)[1:20] # feature names

# create the vectors of the results
pvalues=numeric(20) # p-values of log-rank test
hazards=numeric(20) # hazard ratio
hazardsl=numeric(20) # hazard ratio left 95% bound
hazardsr=numeric(20) # hazard ratio right 95% bound

for(i in 1:20){ # for every island
  
  # bin the classes according to the second classification (quartiles)
  cutted = cut(dataset[,i], breaks=c(0, quantile(dataset[,i],0.25), quantile(dataset[,i],0.75), 1), labels=c("High DNAm", "Medium DNAm", "Low DNAm"))
  
  # select the data only in the first and last quartile
  dataset_quartiles=dataset[cutted!="Medium DNAm",]
  cutted=cutted[cutted!="Medium DNAm"]
  
  # fit the model
  fit_kaplan = survfit(Surv(time.to.disease, event) ~ cutted, data = dataset_quartiles)
  
  # plot the Kaplan-Meier curves
  print(ggsurvplot(fit_kaplan, conf.int = T, risk.table = TRUE,
                   risk.table.col = "strata", surv.median.line = "hv",
                   ggtheme = theme_bw(base_size=18), break.time.by=3.5,
                   ylab = c("Probability"), xlab = c("Years"), 
                   legend.labs=c("Low DNAm","High DNAm"),
                   legend.title="Methylation class", palette=c("red","red4"), 
                   title=paste("Kaplan-Meier Curves for island", names[i]),
                   pval=T))
  
  # compute the hazard ratio confidence interval
  fit_cox = coxph(Surv(time.to.disease, event) ~ cutted, data = dataset_quartiles)
  hazards[i]=exp(coef(fit_cox))[2]
  hazardsl[i]=exp(confint(fit_cox))[2,1]
  hazardsr[i]=exp(confint(fit_cox))[2,2]
  
  # compute the p-value of the log-rank test
  logrank=survdiff(Surv(time.to.disease, event) ~ cutted, data = dataset_quartiles)
  pvalues[i]=pchisq(logrank$chisq, length(logrank$n)-1, lower.tail = FALSE)
}

pvalues #2.574219e-08 8.415108e-06 6.523915e-02 6.850307e-06 5.010730e-04 2.440224e-08 7.004813e-10 8.027209e-07 4.591239e-03 1.079750e-04 8.859561e-07 8.410558e-03 2.670979e-09 1.024563e-09 5.997332e-11 6.122547e-04 7.224611e-05 9.034324e-08 1.216620e-06 9.238797e-10

hazards # 2.833717 2.255000 1.400715 2.253551 1.881367 2.927181 3.064934 2.472790 1.674424 2.010382 2.441011 1.619150 3.039483 3.200802 3.547310 1.858603 2.044721 2.680614 2.539591 3.117682

hazardsl # 1.9370454 1.5633540 0.9771863 1.5679453 1.3109871 1.9797973 2.1145973 1.7072988 1.1679278 1.4026481 1.6932587 1.1280677 2.0749250 2.1687251 2.3795991 1.2969926 1.4268967 1.8441695 1.7235136 2.1325750

hazardsr # 4.145464 3.252637 2.007809 3.238947 2.699907 4.327913 4.442367 3.581499 2.400573 2.881432 3.518975 2.324015 4.452430 4.724034 5.288036 2.663396 2.930055 3.896436 3.742077 4.557842