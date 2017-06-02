############################################################
####### Rashan Smith - Global Terrorism Data Project #######
## This project will use Global Terrorism Data to predict ##
### The form of terrorism a country is likely to ###########
####                  encounter                 ############
############################################################



##################################################
######## Install Packages/Load Libraries #########
##################################################

install.packages("Lock5Data")
install.packages("ggplot2")
install.packages("corrplot")
install.packages("aplpack")
install.packages("modes")
install.packages("googlevis")
install.packages("caret")
install.packages("mlbench")
install.packages("FSelector")
install.packages("rpart")
install.packages("mice")

install.packages("car")
install.packages("lattice")
install.packages("Hmisc")
install.packages("caret")
install.packages("RWeka")
install.packages("e1071")

library(FSelector)
library(rpart)
library(corrplot)
library(caret)
library(ggplot2)
library(mlbench)
library(e1071)

library(car)
library(lattice)
library(Hmisc)
library(caret)
library(RWeka)


################# Load Terrorism Data ####################
globalterrorismdata <- read.csv("globalterrorismdata.csv")



####################################
#  Data Understanding              #
#                                  #
# Visualization and Summarization  #
####################################

# Summarize the features
summary(globalterrorismdata) 
str(globalterrorismdata)

# take a random sample of size 10000 from a dataset mydata
# sample without replacement
mysample <- globalterrorismdata[sample(1:nrow(globalterrorismdata), 10000,
                                       replace=FALSE),] 


### Check for missing data (to help eliminate amount of features from the start) ###
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(mysample,2,pMiss)
apply(mysample,1,pMiss)
sink("missingValues.txt")

subglobaldata <- mysample[ c(2, 3, 4, 8, 10, 16, 23, 26, 27, 28, 29, 35, 41, 64, 69, 84 )]
subglobaldata2 <- mysample[ c(2, 3, 4, 8, 10,14, 15, 16, 20, 23, 26, 27, 28, 29, 35, 37, 41, 64, 69, 84 )]



################################################################
## Select features for global terrorism data analysis #
################################################################
result <- cfs(region ~ ., subglobaldata)

# Obtain the selected subset of features
ce <- as.simple.formula(result, "region")

# Display the selected subset of features
print(ce)

# ggplot2 will be used for plots
require(ggplot2)
plot(subglobaldata, main="Scatter Plot Matrix for GTD Data")

# Bar Plot of Regions
ggplot(subglobaldata, aes(x = region)) + geom_bar()

# Bar Plot of Years
ggplot(subglobaldata, aes(x = iyear)) + geom_bar()
      
# Bar Plot of Attack Types
ggplot(subglobaldata, aes(x = attacktype1)) + geom_bar()


# Plot a histogram of region_text Showing probabilities rather than frequencies
hist(subglobaldata$region, prob=TRUE, main="Histogram of Regions")

# Correlation Plot
# plotting the correlation plot of three features
corr3 <- cor(subglobaldata)
corrplot(corr3, method="shade")


# Combine a histogram with a density curve and "rug" showing 
# the data value distribution
# Add a Quantile-Quantile (Q-Q) plot to check sepal length 
# distribution against a normal distribution
op <- par(mfrow=c(1,2))
hist(subglobaldata$region, prob=TRUE, main="Histogram of\nRegion", ylim=0:1)
lines(density(subglobaldata$region, na.rm=TRUE))
rug(jitter(subglobaldata$region))
qqPlot(subglobaldata$region, main="Normal QQ Plot of\nRegion")
par(op)

# Combine a boxplot of weaptype1 with a rug, again to show the 
# distribution of values, and a dashed line showing the mean
boxplot(subglobaldata$attacktype1, main="Box plot of\nIris Weapon Type")
rug(jitter(subglobaldata$attacktype1), side=2)
abline(h=mean(subglobaldata$attacktype1, na.rm=TRUE), lty=2)


# Use the lattice library to create a 
# boxplot of sepal length conditioned on species
bwplot(attacktype1 ~ region, data=subglobaldata, ylab="attacktype1", xlab="Region")

# Use the lattice and Hmisc libraries to create a box 
# percentile plot of sepal length conditioned on species
bwplot(targtype1 ~ region, data=subglobaldata, 
       panel=panel.bpplot, probs=seq(.01, .49, by=.01),
       datadensity=TRUE,
       ylab="Attack Type", xlab="Region")




###########################################################
## Data Preparation                                       #
##                                                        #
## Includes feature selection and transformations such as #
## discretization, normalization, standardization...      #
###########################################################


##################################################
############   Feature Selection    ##############
##################################################


# Use random forest to choose the most important attributes (e.g. decision tree)
# Inclding sort_group which has a 1.0 correspondence with sort_group_name
subglobaldata.rf.scores <- random.forest.importance(region ~ ., subglobaldata)

# Display the feature scores
print(subglobaldata.rf.scores)

# Show the features with significantly higher importance
cutoff.biggest.diff(subglobaldata.rf.scores)

# Show the top k important features
cutoff.k(subglobaldata.rf.scores, k = 3)

# Show the top k% of import features
cutoff.k.percent(subglobaldata.rf.scores, 0.4)


#####################################################################
## Select features for Terrorism data using correlation and entropy #
#####################################################################
result <- cfs(region~ ., subglobaldata)

# Obtain the selected subset of features
ce <- as.simple.formula(result, "region")

# Display the selected subset of features
print("Entropy")
print(ce)
print("................")


################################################
## MODELING                                    #
##                                             #
## Decision Tree, Rule Set and SVM             #
################################################

# Creating Subset from Actual Data 
subData <- subglobaldata[ c(1, 5, 11, 12, 16 )]

subData$iyear <- factor(subData$iyear)
subData$region <- factor(subData$region)
subData$attacktype1 <- factor(subData$attacktype1)
subData$targtype1 <- factor(subData$targtype1)
subData$weaptype1 <- factor(subData$weaptype1)


# Split terrorism data into train and test sets
set.seed(1)
trainSet <- createDataPartition(subData$region, p=.7)[[1]]
subData.train <- subData[trainSet,]
subData.test <- subData[-trainSet,]

#########################################################################
# Build a DECISION TREE for region using C4.5 (Weka's J48 implementation)
#########################################################################
gtd.model.nom <- J48(region ~ ., data=subData.train)

# View details of the constructed tree
# Be sure you understand each of the reported measures
# (how it is calculated and what it means)
summary(gtd.model.nom )

# Plot the decision tree
plot(gtd.model.nom )

#########################################################################
# Build a Rule Set Using RIPPER
# Remember that RIPPER creates a default rule for the majority class
# and then creates rules to cover the other classes
#########################################################################

# Build the rule set
gtd.model.rules <- JRip(region ~ ., data=subData.train)

# Display the rule set
print(gtd.model.rules)


#########################################################################
# SVM 
#########################################################################
gtd.model.svm <- svm(region~., data=subData.train)

# Plot the  model
plot(gtd.model.svm, subData.train)

#########################################################################
# Logistic Regression
#########################################################################
#subData.logmodel.reg <- glm(region~ ., data=subData.train, family=binomial(link="logit"))
#summary(subData.logmodel.reg)



###############
## Evaluation #
###############

###################### Evaluate Rules ####################
# Create predictions from the rule set using the test set
gtd.predict.rules  <- predict(gtd.model.rules , subData.test)

# Calculation of performance for nominal values uses a confusion matrix
# and related measures.
gtd.eval.rules  <- confusionMatrix(gtd.predict.rules , subData.test$region)

print(gtd.eval.rules)


######## Evaluate Decision Tree Model ##########
gtd.predict.dec <- predict(gtd.model.nom, subData.test,  type="c")

###### Decision Tree Confusion Matrix #####
subData.predict.dec <- confusionMatrix(gtd.predict.dec, subData.test$region)
print(subData.predict.dec)


############# Evaluate SVM model #####################
gtd.eval.svm <- predict(gtd.model.svm, na.omit(subData.test))

############ SVM Confusion Matrix ######################
gtd.eval.svm.conMat <- confusionMatrix(gtd.eval.svm , na.omit(subData.test)$region)

print(gtd.eval.svm.conMat)

########### Evaluate Rules Model ######################


######################### Logistic Regression work on levels ######################
#subData.logpredict.reg <- predict(subData.logmodel.reg, subData.test, type="response")

# Confusion Matrix
#ubData.predict.nom <- confusionMatrix(subData.predict.reg, subData.test$region)
#print(subData.predict.nom)

