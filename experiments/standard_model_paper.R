# load the packages
library(survival)

# load dataset
dataset <- read.csv(file = 'Dataset_Island_means_time.csv')
dataset['event']=rep(2,248) # all cases
head(dataset)

# load the island names
dataset2 <- read.csv(file = 'Dataset_for_Cox.csv', check.names = FALSE)
islands=colnames(dataset2[2:3808])

# initialize the results
coefs=numeric(3807)
pvalues=numeric(3807)
z=numeric(3807)
concordance=numeric(3807)

for (i in 2:3808){ # for each island
  
  # select the island and normalize it
  variable=dataset[,i]
  variable=normalize(variable)
  
  # fit the cox model
  cox <- coxph(Surv(time.to.disease, event) ~ variable, data = dataset)
  
  # extract the relevant quantities
  info=summary(cox)
  coef_info=info$coefficients
  
  # append in the results
  coefs[i-1]=coef_info[1]
  pvalues[i-1]=coef_info[5]
  z[i-1]=coef_info[4]
  concordance[i-1]=info$concordance[1]
}

# create the matrix of the results
results=data.frame(CpG_Island=islands, pvalues=pvalues, stattest=abs(z), coefabs=abs(coefs), concordance=concordance)
saveRDS(results, "pvalues_standard.rds")
write.csv(results,'pvalues_standard.csv')
