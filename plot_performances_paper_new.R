# load the package
library(ggplot2)
library(dplyr)

# names of the models
names=c('No Pretraining NN (128-16)', 'No Pretraining NN (256-16)', 'No Pretraining NN (512-16)', 'No Pretraining NN (1024-16)', 'No Pretraining NN (128-32)', 'No Pretraining NN (256-32)', 'No Pretraining NN (512-32)', 'No Pretraining NN (1024-32)', 'XGBoost (128)', 'XGBoost (256)', 'XGBoost (512)', 'XGBoost (1024)', 'CoxPH (128)', 'CoxPH (256)', 'CoxPH (512)', 'CoxPH (1024)', 'Deep Surv. NN (128-16)', 'Deep Surv. NN (256-16)', 'Deep Surv. NN (512-16)', 'Deep Surv. NN (1024-16)', 'Deep Surv. NN (128-32)', 'Deep Surv. NN (256-32)', 'Deep Surv. NN (512-32)', 'Deep Surv. NN (1024-32)')

# report the results of the stability performances
means=c(0.613, 0.600, 0.609, 0.590, 0.605, 0.622, 0.620, 0.628, 0.697, 0.683, 0.754, 0.833, 0.593, 0.621, 0.601, 0.620, 0.669, 0.631, 0.606, 0.648, 0.609, 0.644, 0.615, 0.605)
lowers=c(0.582, 0.562, 0.571, 0.553, 0.576, 0.589, 0.571, 0.589, 0.660, 0.643, 0.726, 0.811, 0.555, 0.579, 0.559, 0.587, 0.632, 0.592, 0.572, 0.611, 0.570, 0.609, 0.575, 0.569)
uppers=c(0.644, 0.638, 0.647, 0.627, 0.634, 0.655, 0.669, 0.667, 0.734, 0.723, 0.782, 0.855, 0.631, 0.663, 0.643, 0.653, 0.707, 0.670, 0.639, 0.685, 0.649, 0.679, 0.656, 0.641)
data <- structure(list(mean  = means, lower = lowers, upper = uppers), 
                  .Names = c("mean", "lower", "upper"), row.names = names, class = "data.frame")

# report the results of the predictive performances
means1=c(0.676, 0.665, 0.673, 0.715, 0.696, 0.686, 0.682, 0.707, 0.693, 0.711, 0.676, 0.678, 0.610, 0.586, 0.628, 0.652, 0.702, 0.710, 0.701, 0.716, 0.698, 0.713, 0.694, 0.712)
lowers1=c(0.660, 0.646, 0.654, 0.696, 0.676, 0.660, 0.656, 0.683, 0.676, 0.692, 0.656, 0.654, 0.578, 0.562, 0.603, 0.635, 0.683, 0.693, 0.682, 0.699, 0.675, 0.694, 0.673, 0.695)
uppers1=c(0.692, 0.684, 0.692, 0.734, 0.716, 0.712, 0.708, 0.731, 0.709, 0.730, 0.696, 0.702, 0.642, 0.612, 0.653, 0.669, 0.722, 0.726, 0.720, 0.733, 0.722, 0.732, 0.716, 0.728)
data1 <- structure(list(mean  = means1, lower = lowers1, upper = uppers1), 
                  .Names = c("mean", "lower", "upper"), row.names = names, class = "data.frame")

# assign the colors according to the type of the model
colors=as.data.frame(c(2,2,2,2,2,2,2,2,3,3,3,3,0,0,0,0,1,1,1,1,1,1,1,1))
colnames(colors)='colors_col'
colors=mutate(colors, cond = case_when(colors_col==2 ~ 'green', colors_col==3 ~ 'red', colors_col==0 ~ 'black', colors_col==1  ~ 'blue'))

# create forest plot of the stability performances
ggplot(data=data, aes(y=seq(1, 24, by=1), x=mean, xmin=lower, xmax=upper, colour=colors$cond), show.legend = FALSE) +
  scale_color_manual(values=c("black", "blue", "red", "green")) +
  geom_point() + 
  geom_errorbarh(height=.3) +
  scale_y_continuous(name = "", breaks=1:nrow(data), labels=names) +
  scale_x_continuous(limits = c(0.4, 0.9)) +
  labs(title='Stability performances', x='KT-Stability', y = 'Model') +
  theme_minimal(base_size=18)

# create forest plot of the predictive performances
ggplot(data=data1, aes(y=seq(1, 24, by=1), x=mean, xmin=lower, xmax=upper, colour=colors$cond), show.legend = FALSE) +
  scale_color_manual(values=c("black", "blue", "red", "green")) +
  geom_point() + 
  geom_errorbarh(height=.3) +
  scale_y_continuous(name = "", breaks=1:nrow(data1), labels=names) +
  scale_x_continuous(limits = c(0.4, 0.9)) +
  labs(title='Predictive performances', x='C-Index', y = 'Model') +
  geom_vline(xintercept=0.5, color='black', linetype='dashed', alpha=.5) +
  theme_minimal(base_size=18)
