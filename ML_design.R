# ==============================================================================
# BIO215 Capstone Project: Machine Learning Pipeline
# Author: Hongyao Yang
# Models: Logistic, ElasticNet, NaiveBayes, RF, SVM, kNN, LightGBM
# Output: Model files (.rds), Comparison Plots, Final Leaderboard
# ==============================================================================

# ==============================================================================
# 0. Environment Configuration & Dependency Management
# ==============================================================================
required_packages <- c("tidyverse", "caret", "ranger", "glmnet", "naivebayes",
                       "class", "e1071", "kernlab", "lightgbm", "pROC",
                       "doParallel", "PRROC", "ggplot2")

# Check and install missing dependencies
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  print(paste("Installing missing dependencies:", paste(new_packages, collapse = ", ")))
  install.packages(new_packages)
}

# Import libraries
library(tidyverse)
library(caret)
library(ranger)      # Random Forest Implementation
library(glmnet)      # Elastic Net Regularization
library(naivebayes)  # Naive Bayes Classifier
library(class)       # k-Nearest Neighbors
library(e1071)       # Support Vector Machines
library(lightgbm)    # Light Gradient Boosting Machine
library(pROC)        # Receiver Operating Characteristic Analysis
library(PRROC)       # Precision-Recall Curve Analysis
library(doParallel)  # Parallel Computing Backend
library(ggplot2)     # Data Visualization

# Initialize parallel processing backend
detected_cores <- detectCores()
worker_count <- if (is.na(detected_cores)) 1 else max(1, detected_cores - 1)
cl <- NULL
if (worker_count > 1) {
  cl <- makePSOCKcluster(worker_count)
  registerDoParallel(cl)
  print(paste("Environment initialization complete.", worker_count, "parallel workers active."))
} else {
  registerDoSEQ()
  print("Environment initialization complete. Sequential processing active.")
}

# ==============================================================================
# 1. Data Loading & Preprocessing
# ==============================================================================
data_path <- "alzheimers_disease_data.csv"
if(!file.exists(data_path)) stop("Error: Dataset not found. Verify file path.")
data <- read.csv(data_path)

# A. Exclusion of Identifiers and High-Leakage Features
# Retaining MMSE and ADL as per project requirements
cols_to_remove <- c("PatientID", "DoctorInCharge", "FunctionalAssessment")
cols_to_remove <- cols_to_remove[cols_to_remove %in% names(data)]
data_clean <- data %>% select(-all_of(cols_to_remove))

print(paste("Data Cleaning: Excluded columns:", paste(cols_to_remove, collapse=", ")))

# B. Feature Engineering
if("SystolicBP" %in% names(data_clean) & "DiastolicBP" %in% names(data_clean)) {
  data_clean$PulsePressure <- data_clean$SystolicBP - data_clean$DiastolicBP
  print("Feature Engineering: Derived variable 'PulsePressure' computed.")
}

# C. Standardization of Column Nomenclature
colnames(data_clean) <- make.names(colnames(data_clean))

# D. Target Variable Transformation
data_clean$Diagnosis <- factor(data_clean$Diagnosis, levels = c(0, 1), labels = c("No", "Yes"))

# E. Categorical Feature Encoding
categorical_cols <- c("Gender", "Ethnicity", "EducationLevel", "Smoking",
                      "FamilyHistoryAlzheimers", "CardiovascularDisease",
                      "Diabetes", "Depression", "HeadInjury", "Hypertension",
                      "MemoryComplaints", "BehavioralProblems", "Confusion",
                      "Disorientation", "PersonalityChanges",
                      "DifficultyCompletingTasks", "Forgetfulness")

for(col in categorical_cols) {
  if(col %in% names(data_clean)) {
    data_clean[[col]] <- as.factor(data_clean[[col]])
  }
}

print(paste("Final Dataset Dimensions:", nrow(data_clean), "observations,", ncol(data_clean), "variables"))

# ==============================================================================
# 2. Dataset Partitioning
# ==============================================================================
set.seed(42)
trainIndex <- createDataPartition(data_clean$Diagnosis, p = 0.8, list = FALSE)
train_set <- data_clean[trainIndex,]
test_set  <- data_clean[-trainIndex,]

# ==============================================================================
# 3. Training Control Configuration (Cross-Validation Protocol)
# ==============================================================================
fitControl <- trainControl(method = "cv",
                           number = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           # Critical: Retain validation predictions for CV metric aggregation
                           savePredictions = "final",
                           allowParallel = TRUE)

# ==============================================================================
# 4. Model Training Pipeline
# ==============================================================================
print("Initiating training pipeline for 7 candidate models...")

# --- Models 1-6: Caret Framework (Automated CV) ---
print("[1/7] Training Logistic Regression...")
model_glm <- train(Diagnosis ~ ., data = train_set, method = "glm", metric = "ROC", trControl = fitControl)

print("[2/7] Training Elastic Net...")
model_enet <- train(Diagnosis ~ ., data = train_set, method = "glmnet", metric = "ROC", trControl = fitControl, tuneLength = 10)

print("[3/7] Training Naive Bayes...")
model_nb <- train(Diagnosis ~ ., data = train_set, method = "naive_bayes", metric = "ROC", trControl = fitControl)

print("[4/7] Training Random Forest...")
model_rf <- train(Diagnosis ~ ., data = train_set, method = "ranger", metric = "ROC", trControl = fitControl, importance = 'impurity')

print("[5/7] Training Support Vector Machine (Radial Basis)...")
model_svm <- train(Diagnosis ~ ., data = train_set, method = "svmRadial", metric = "ROC", preProcess = c("center", "scale"), trControl = fitControl)

print("[6/7] Training k-Nearest Neighbors...")
model_knn <- train(Diagnosis ~ ., data = train_set, method = "knn", metric = "ROC", preProcess = c("center", "scale"), trControl = fitControl)

# --- Model 7: LightGBM (Final Model + Manual 5-Fold CV) ---
print("[7/7] Training LightGBM (Final Model + Cross-Validation)...")

# A. Train Final Model (Full Training Set for Independent Testing)
train_label_lgb <- ifelse(train_set$Diagnosis == "Yes", 1, 0)
train_matrix <- as.matrix(sapply(train_set %>% select(-Diagnosis), as.numeric))
test_matrix  <- as.matrix(sapply(test_set %>% select(-Diagnosis), as.numeric))

dtrain <- lgb.Dataset(data = train_matrix, label = train_label_lgb)

# LightGBM Hyperparameters
params_lgb <- list(objective = "binary", metric = "auc", learning_rate = 0.1, num_leaves = 31, verbose = -1)

model_lgb <- lgb.train(
  params = params_lgb,
  data = dtrain,
  nrounds = 100,
  verbose = -1
)

# B. Execute Manual 5-Fold CV (To ensure comparability with Caret models)
print("Executing Manual 5-Fold Cross-Validation for LightGBM consistency...")
set.seed(42)
lgb_folds <- createFolds(train_set$Diagnosis, k = 5, list = TRUE, returnTrain = FALSE)
lgb_cv_probs <- numeric(nrow(train_set)) # Container for validation set predictions

for(fold_idx in seq_along(lgb_folds)) {

  val_idx <- lgb_folds[[fold_idx]]
  train_idx <- setdiff(seq_len(nrow(train_set)), val_idx)

  # Fold Data Preparation
  dtrain_fold <- lgb.Dataset(data = train_matrix[train_idx, ], label = train_label_lgb[train_idx])

  # Train Fold Model
  model_cv <- lgb.train(
    params = params_lgb,
    data = dtrain_fold,
    nrounds = 100,
    verbose = -1
  )

  # Validation Prediction (Out-of-Fold)
  lgb_cv_probs[val_idx] <- predict(model_cv, train_matrix[val_idx, ])
}

# ==============================================================================
# 5. Results Compilation & Leaderboard Generation
# ==============================================================================
if (!is.null(cl)) stopCluster(cl)
registerDoSEQ()

print("Calculating performance metrics...")

# --- Helper 1: Metric Calculation Routine ---
get_pred_metrics <- function(probs, truth_label) {
  truth_num <- ifelse(truth_label == "Yes", 1, 0)
  preds_label <- ifelse(probs > 0.5, "Yes", "No")

  roc_obj <- roc(truth_num, probs, quiet = TRUE)
  pr_obj <- pr.curve(scores.class0 = probs, weights.class0 = truth_num, curve = FALSE)
  cm <- confusionMatrix(factor(preds_label, levels = c("No", "Yes")),
                        factor(truth_label, levels = c("No", "Yes")))

  return(c(
    AUC = as.numeric(auc(roc_obj)),
    PRC = pr_obj$auc.integral,
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    Accuracy = as.numeric(cm$overall["Accuracy"])
  ))
}

# --- Helper 2: Caret CV Metric Extraction ---
get_cv_metrics <- function(model) {
  if(is.null(model$pred)) return(rep(NA, 5))
  best_preds <- model$pred
  return(get_pred_metrics(best_preds$Yes, best_preds$obs))
}

model_list_caret <- list(
  Logistic = model_glm, ElasticNet = model_enet, NaiveBayes = model_nb,
  RandomForest = model_rf, SVM = model_svm, kNN = model_knn
)

# --- A. Generate Training Leaderboard (Aggregated 5-Fold CV) ---
print("Aggregating 5-Fold Cross-Validation Results...")
# 1. Caret Models
results_cv <- lapply(names(model_list_caret), function(name) {
  stats <- get_cv_metrics(model_list_caret[[name]])
  c(name, stats)
})
leaderboard_train <- do.call(rbind, results_cv) %>% as.data.frame()

# 2. LightGBM (Incorporating Manual CV probabilities)
lgb_cv_stats <- get_pred_metrics(lgb_cv_probs, train_set$Diagnosis)
leaderboard_train <- rbind(leaderboard_train, c("LightGBM", lgb_cv_stats))

# Formatting
colnames(leaderboard_train) <- c("Model", "AUC", "PRC", "Sensitivity", "Specificity", "Accuracy")
leaderboard_train <- leaderboard_train %>%
  mutate_at(vars(AUC, PRC, Sensitivity, Specificity, Accuracy), as.numeric) %>%
  arrange(desc(AUC))

write.csv(leaderboard_train, "leaderboard_train_cv.csv", row.names = FALSE)

# --- B. Generate Test Leaderboard (Independent Test Set) ---
print("Evaluating on Independent Test Set...")
# 1. Caret Models
results_test <- lapply(names(model_list_caret), function(name) {
  probs <- predict(model_list_caret[[name]], test_set, type = "prob")$Yes
  stats <- get_pred_metrics(probs, test_set$Diagnosis)
  c(name, stats)
})
leaderboard_test <- do.call(rbind, results_test) %>% as.data.frame()

# 2. LightGBM (Using final model trained on full train set)
lgb_test_probs <- predict(model_lgb, test_matrix)
lgb_test_stats <- get_pred_metrics(lgb_test_probs, test_set$Diagnosis)
leaderboard_test <- rbind(leaderboard_test, c("LightGBM", lgb_test_stats))

# Formatting
colnames(leaderboard_test) <- c("Model", "AUC", "PRC", "Sensitivity", "Specificity", "Accuracy")
leaderboard_test <- leaderboard_test %>%
  mutate_at(vars(AUC, PRC, Sensitivity, Specificity, Accuracy), as.numeric) %>%
  arrange(desc(AUC))

write.csv(leaderboard_test, "leaderboard_test_final.csv", row.names = FALSE)

# --- D. Output Results ---
print("============================================================")
print("[TRAIN] 5-Fold Cross-Validation Performance Summary:")
print("(Methodology: LightGBM uses manual CV; others use Caret CV)")
print(leaderboard_train, row.names = FALSE)
print("------------------------------------------------------------")
print("[TEST] Independent Test Set Evaluation:")
print(leaderboard_test, row.names = FALSE)
print("============================================================")

# --- Visualization ---
print("Generating model comparison visualizations...")
ggplot(leaderboard_test, aes(x = reorder(Model, -AUC), y = AUC)) +
  geom_bar(stat = "identity", fill = "steelblue", width = 0.7) +
  geom_text(aes(label = round(AUC, 3)), vjust = -0.5, size = 3.5) +
  scale_y_continuous(limits = c(0, 1.1), expand = c(0,0)) +
  theme_minimal() +
  labs(title = "Model Comparison: Area Under Curve (Test Set)", x = "Model", y = "AUC Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
        plot.margin = margin(1, 1, 1, 1, "cm"))

# ==============================================================================
# 6. Model Archival
# ==============================================================================
print("Archiving optimal model artifacts...")
saveRDS(model_rf, "final_model_rf.rds")
saveRDS(model_enet, "final_model_enet.rds")
lgb.save(model_lgb, "final_model_lgb.txt")

print("Pipeline execution terminated successfully.")
