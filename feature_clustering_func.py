# function to transform a scaled dataset according to a given feature clustering and join the non-genomic variables
def transform_clustering(Dataset_scaled, clusterer, Dataset2):

    # apply feature clustering
    Dataset_reduced = clusterer.transform(Dataset_scaled)

    # transform in the complete dataset
    Dataset_final = pd.DataFrame(Dataset_reduced, index=Dataset2.index)
    Dataset_final = Dataset_final.join(Dataset2[
                                           ['time.to.disease', 'study', 'age.recr', 'bmi', 'sex', 'smoking', 'alcohol',
                                            'education', 'phys.act', 'dietary.qual', 'class', 'match.id',
                                            'time.binned']])

    return Dataset_final



# function to create the dataset of the results for the shap models
def construct_dataset_results(path, name, columns, lab, n_clusters, latent_dim=0):

    # load the global shaps value for the correct reference
    if latent_dim==0:
        shaps = np.load(os.path.join(path + name + str(n_clusters) + '.npy'))
    else:
        shaps = np.load(os.path.join(path + name + str(n_clusters) + "_" + str(latent_dim) + '.npy'))

    # calculate the mean global shap value across the different splits
    mean_shaps = []
    for i in range(n_clusters):
        cur = []
        for j in range(10):
            cur.append(shaps[j][i])
        mean_shaps.append(np.mean(cur))

    # order the features
    mean_shaps1 = np.array(mean_shaps)
    I = np.argsort(-mean_shaps1)
    I = I.astype('str')

    # create the dataframe
    d = {'CpG_Island': columns, 'Clustered_Feature': lab}
    df = pd.DataFrame(data=d)
    df['Importance'] = np.zeros(3807)
    df['Ranking'] = np.zeros(3807)
    j = 1
    for i in I:
        df['Ranking'][df['Clustered_Feature'] == i.astype(int)] = j
        df['Importance'][df['Clustered_Feature'] == i.astype(int)] = mean_shaps[i.astype(int)]
        j = j + 1
    df['Ranking'] = df['Ranking'].astype(int)

    return df



# function to plot the importance of each island
def plot_results(df):

  # array for mixing colors
  shuffle=np.array([1,15,20,2,6,18,22,5,11,21,13,7,14,4,12,19,10,3,9,17,8,16])

  # order the islands correctly
  df['chromosome']=df.CpG_Island.str.extract('(\d+)').astype(int)
  df['Index']=df.index.to_series()
  df1=df.sort_values(by=['chromosome', 'Index'])

  # calculate the positioning of the xlabels in the plot
  unique, counts = np.unique(df['chromosome'], return_counts=True)
  position=np.empty(22)
  cur=0
  for i in range(22):
    position[i]=cur+counts[i]/2
    cur=cur+counts[i]

  # scatter plot
  plt.style.use('default')
  plt.rcParams.update({'font.size': 12})
  plt.figure(figsize=(20,8))
  plt.scatter(df1['CpG_Island'], df1['Importance'], c=shuffle[df1['chromosome']-1], cmap='nipy_spectral', s=10)
  plt.xlabel('Chromosome')
  plt.ylabel('Importance')
  ticks = position
  labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
  plt.xticks(ticks,labels=labels,rotation=90)
  plt.show()



# function to plot the importance of each island according to the standard EWAS approach
def plot_results_cox(path, name, mode):

    # load the pvalues of the standard EWAS approach
    pvals=pd.read_csv(os.path.join(path + name + '.csv'), index_col="Unnamed: 0")

    # creation of the variables for the result
    pvals['log'] = -np.log10(pvals['pvalues'])
    pvals['adjusted'] = multipletests(np.array(pvals['pvalues']), method='bonferroni')[1]
    pvals['log_adjusted'] = -np.log10(pvals['adjusted'])

    # array for mixing colors
    shuffle = np.array([1, 15, 20, 2, 6, 18, 22, 5, 11, 21, 13, 7, 14, 4, 12, 19, 10, 3, 9, 17, 8, 16])

    # order the islands correctly
    pvals['chromosome'] = pvals.CpG_Island.str.extract('(\d+)').astype(int)
    pvals['Index'] = pvals.index.to_series()
    df1 = pvals.sort_values(by=['chromosome', 'Index'])

    # calculate the positioning of the xlabels in the plot
    unique, counts = np.unique(df1['chromosome'], return_counts=True)
    position = np.empty(22)
    cur = 0
    for i in range(22):
        position[i] = cur + counts[i] / 2
        cur = cur + counts[i]

    # scatter plot the logarithm of the pvalues
    if mode == "standard":
        plt.style.use('default')
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.scatter(df1['CpG_Island'], df1['log'], c=shuffle[df1['chromosome'] - 1], cmap='nipy_spectral', s=10)
        plt.xlabel('Chromosome')
        plt.ylabel('- Log10 (p-value)')
        ticks = position
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        plt.xticks(ticks, labels=labels, rotation=90)
        ax.hlines(y=-np.log10(0.05), xmin=0, xmax=3807, linewidth=3, color='black', linestyles='--')
        plt.show()

    # scatter plot the logarithm of the bonferroni adjusted pvalues
    if mode == "bonferroni":
        plt.style.use('default')
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.scatter(df1['CpG_Island'], df1['log_adjusted'], c=shuffle[df1['chromosome'] - 1], cmap='nipy_spectral', s=10)
        plt.xlabel('Chromosome')
        plt.ylabel('- Log10 (p-value Bonferroni adjusted)')
        ticks = position
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        plt.xticks(ticks, labels=labels, rotation=90)
        ax.hlines(y=-np.log10(0.05), xmin=0, xmax=3807, linewidth=3, color='black', linestyles='--')
        plt.show()



# function to plot the histogram of the numerosity of the clusters for each clustering dimension
def plot_granularity(counts_list, labels):

    # extract the number of islands in each feature
    counts0 = counts_list[0]
    counts1 = counts_list[1]
    counts2 = counts_list[2]
    counts3 = counts_list[3]

    # define the weights of each dimension to plot the percentage (same scale)
    w0 = np.empty(counts0.shape)
    w0.fill(1 / counts0.shape[0])
    w1 = np.empty(counts1.shape)
    w1.fill(1 / counts1.shape[0])
    w2 = np.empty(counts2.shape)
    w2.fill(1 / counts2.shape[0])
    w3 = np.empty(counts3.shape)
    w3.fill(1 / counts3.shape[0])
    bins = np.arange(0, 67, 6)

    # histogram of the 4 dimensionalities
    plt.style.use('default')
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(16, 8))
    plt.hist(
        [np.clip(counts0, bins[0], bins[-1]), np.clip(counts1, bins[0], bins[-1]), np.clip(counts2, bins[0], bins[-1]),
         np.clip(counts3, bins[0], bins[-1])], weights=[w0, w1, w2, w3], label=labels, bins=bins)
    xlabels = ['1-6', '7-12', '13-18', '19-24', '25-30', '31-36', '37-42', '43-48', '49-54', '55-60', '60+']
    N_labels = len(xlabels)
    plt.xlim([0, 66])
    plt.xticks(6 * np.arange(N_labels) + 3, labels=xlabels)
    plt.legend(loc='upper right')
    plt.title("Granularity with different clustering dimensions")
    plt.xlabel("Number of elements in a cluster")
    plt.ylabel("Percentage")
    plt.show()