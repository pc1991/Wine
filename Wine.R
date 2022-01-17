library(faraway)
library(readr)
library(mlbench)
library(caret)
library(corrplot)

wine <- read_csv("WineQT.csv")
spec(wine)
head(wine, n = 20)

validationIndex <- createDataPartition(wine$Id, p = .8, list = FALSE)
validation <- wine[-validationIndex,]
liquor <- wine[validationIndex,]

dim(liquor)
sapply(liquor, class)

x <- liquor[,1:12]
y <- liquor[,13]

#jittered scatter plot matrix
winejitter <- sapply(liquor[,1:12], jitter)
pairs(winejitter, names(liquor[,1:12]), col = house$price)

#Correlation Plot#
correlations <- cor(liquor[,1:12])
corrplot(correlations, method = "circle")

#Run the algorithms using 10-fold cross-validation#
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "RMSE"

#Linear Model#
set.seed(7)
fit.lm <- train(Id ~ ., data = liquor, method = "lm", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Generalized Linear Model#
set.seed(7)
fit.glm <- train(Id ~ ., data = liquor, method = "glm", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Cubist#
set.seed(7)
fit.cubist <- train(Id ~ ., data = liquor, method = "cubist", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Partial Least Squares#
set.seed(7)
fit.pls <- train(Id ~ ., data = liquor, method = "pls", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Random Forest#
set.seed(7)
fit.rf <- train(Id ~ ., data = liquor, method = "rf", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Elasticnet#
set.seed(7)
fit.enet <- train(Id ~ ., data = liquor, method = "enet", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#Gaussian Process#
set.seed(7)
fit.gaussprLinear <- train(Id ~ ., data = liquor, method = "gaussprLinear", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

#k-Nearest Neighbors#
set.seed(7)
fit.knn <- train(Id ~ ., data = liquor, method = "knn", metric = metric, preProc = c("BoxCox"), trControl = trainControl)

outcome <- resamples(list(LM = fit.lm, GLM = fit.glm, CUBIST = fit.cubist, PLS = fit.pls, RF = fit.rf, ENET = fit.enet, GAUSSIAN = fit.gaussprLinear, KNN = fit.knn))
summary(outcome)
dotplot(outcome)

#Cubist Wins#
library(Cubist)

print(fit.cubist)
plot(fit.cubist)

#Tune the Cubist algorithm#
trainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
metric <- "RMSE"
set.seed(7)
grid <- expand.grid(committees = seq(15, 25, by = 1), neighbors = c(3, 5, 7))
tune.cubist <- train(Id ~ ., data = liquor, method = "cubist", metric = metric, preProc = c("BoxCox"), tuneGrid = grid, trControl = trainControl)
print(tune.cubist)
plot(tune.cubist)

#Prepare the data transform using training data
set.seed(7)
tX <- sample(1:nrow(liquor), floor(.8*nrow(liquor)))
p <- c("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality")
tXp <- liquor[tX, p]
tXr <- liquor$Id[tX]
fM <- cubist(x = tXp, y = tXr, commitees = 25, neighbors = 7)
summary(fM)
predictions <- predict(fM, tXp)

#Compute the RMSE & R^2#
rmse <- sqrt(mean((predictions - tXr)^2))
r2 <- cor(predictions, tXr)^2
print(rmse) #365.7767
print(r2) #0.3782792