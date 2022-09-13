load("EPIC_dnam_data.rda")
load("cgs_EZH2.rda")
# dnam is a table containing the beta values for each CpG site (rows = sites, columns = patients)
# cgs_EZH2 is a list of all the sites in the EZH2 gene
# samples is a table containing all the non-genomic information of samples (rows = patients, columns = features)
dnam_filtered = dnam[rownames(dnam) %in% cgs_EZH2,] # select only EZH2 protein sites
Dataset_raw = merge(samples, t(dnam_filtered), by=0) # join all the information about patients
Dataset = data.frame(Dataset_raw[,-1], row.names=Dataset_raw[,1]) # assign the row names
write.csv(Dataset,"Dataset_EZH2.csv") # save


# annotations contains all the information about CpG sites (chromosomes, islands...)
# select only EZH2 protein sites from annotations and dnam
annotations_filtered = annotations[rownames(annotations) %in% cgs_EZH2,]
Dataset_EZH2 = t(dnam[rownames(dnam) %in% cgs_EZH2,])

# extract the islands names
Islands = unique(annotations_filtered$HMM_Island)
Islands = Islands[-8] # do not consider the sites not belonging to an island

for(i in 1:length(Islands)){ # for each island
  
  # select the sites in the island
  current_sites=rownames(annotations_filtered[which(annotations_filtered$HMM_Island==Islands[i]),]) 
  
  # create a new variable as the mean of the sites of the island
  x=as.matrix(Dataset_EZH2[,colnames(Dataset_EZH2) %in% current_sites])
  D=rowMeans(x)
  
  # add to the final dataset
  if (i==1){
    new_dataset = data.frame(D)
  }else{
    new_dataset=cbind(new_dataset,D)
  }
}

# assign names to rows and columns
rownames(new_dataset) = rownames(Dataset_EZH2)
colnames(new_dataset) = Islands

# save the dataset
write.csv(new_dataset,"Dataset_Islands_Means.csv")
