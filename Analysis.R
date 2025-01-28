#Libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(haven)
library(mice)
library(caret)
library(randomForest)
#Set working directory
#setwd("C:/Users/ciros/Dropbox/Introduction to ML")
setwd("C:/Users/Frank/Dropbox/Introduction to ML")
#Load dataset
dataset <- read_dta("Data Generation/Output/final.dta")

# --------------------------
# DATA PREPARATION
# --------------------------

# Data conversion
dataset_processed <- dataset %>%
  # Convert character columns to factors
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.labelled), as_factor)) %>%
  mutate(across(where(~ inherits(.x, "Date")), as.numeric)) # Convert Date to numeric
# Remove unnecessary variables
dataset_processed <- dataset_processed[, !(names(dataset_processed) %in% c("lang", "nat_region", "nat", "judgename"))]

dataset_processed <- dataset_processed %>%
  mutate(attorney_pres = ifelse(attorney_pres == 0, "NO", "YES"), # Converti in "NO" e "YES"
         attorney_pres = factor(attorney_pres)) # Trasforma in fattore

# --------------------------
# MISSING DATA IMPUTATION
# --------------------------
sum(is.na(dataset_processed))

methods <- c(
  "polyreg",  # dec_code (nominal categorical)
  "pmm",      # dec_dur (continuous)
  "pmm",      # nbr_of_charges (discrete numeric)
  "polyreg",  # base_city_code (nominal categorical)
  "pmm",      # nbr_of_applications (discrete numeric)
  "pmm",      # nbr_of_schedules (continuous)
  "polyreg",  # custody (nominal categorical)
  "pmm",      # crim_ind (continuous)
  "pmm",      # nbr_of_appeals (discrete numeric)
  "logreg",   # attorney_pres (binary categorical)
  "logreg",   # ij_gender (binary categorical)
  "pmm",      # yr_app (continuous)
  "logreg",   # party (binary categorical)
  "polyreg"  # region_grouped (nominal categorical)
)

# Run mice
mice_data <- mice(dataset_processed, method = methods, m = 1, maxit = 1, seed = 123)
dataset_imputed <- complete(mice_data)
sum(is.na(dataset_imputed))

# Last changes
dataset_definitive <- dataset_imputed %>%
  mutate(
    years_on_bench = as.numeric(format(Sys.Date(), "%Y")) - yr_app,
  ) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.labelled), as_factor)) %>%
  mutate(across(where(~ inherits(.x, "Date")), as.numeric))

dataset_definitive <- dataset_definitive[, !(names(dataset_definitive) %in% c("yr_app"))]

# Save definitive dataset for Neural Network analysis 
save(dataset_definitive, file = "Data Generation/Output/final.Rdata")
load("Data Generation/Output/final.Rdata")

# --------------------------
# TRAIN/TEST SPLIT
# --------------------------

set.seed(123)
train_index <- createDataPartition(dataset_definitive$dec_code, p = 0.8, list = FALSE)
train_data <- dataset_definitive[train_index, ]
test_data <- dataset_definitive[-train_index, ]

# --------------------------
# RANDOM FOREST
# --------------------------

# Train the Random Forest model
rf_model <- randomForest(dec_code ~ ., data = train_data, importance = TRUE, ntree = 50)

# Evaluate the model on the test set
rf_predictions <- predict(rf_model, test_data)
confusionMatrix(rf_predictions, test_data$dec_code)

# Extract variable importance
importance_data <- importance(rf_model)
mean_decrease_gini <- importance_data[, "MeanDecreaseGini"]
sorted_gini <- sort(mean_decrease_gini, decreasing = FALSE)

par(mar = c(5, 8, 4, 2))  # Increase left margin space (second value)

# Create a barplot for MeanDecreaseGini in sorted order
barplot(sorted_gini, 
        main = "RF Variable Importance", 
        horiz = TRUE,              # Horizontal bars
        las = 1,                   # Rotate labels for readability
        col = "skyblue",           # Bar color
        xlab = "Mean Decrease Gini", 
        cex.names = 0.8)           # Adjust label size

# --------------------------
# OUT OF BAG ERROR
# --------------------------

# Define a sequence of ntree values to test
ntree_values <- seq(25, 200, by = 25)
# Initialize a vector to store out-of-bag (OOB) error for each model
oob_errors <- numeric(length(ntree_values))

# Loop over ntree values, train the model and record the OOB error
for (i in seq_along(ntree_values)) {
  rf_model <- randomForest(dec_code ~ ., data = train_data, ntree = ntree_values[i], importance = TRUE)
  oob_errors[i] <- rf_model$err.rate[ntree_values[i], "OOB"]
}

# Create a data frame for plotting
plot_data <- data.frame(ntree = ntree_values, OOB_Error = oob_errors)
# Plot OOB Error vs Number of Trees
ggplot(plot_data, aes(x = ntree, y = OOB_Error)) +
  geom_line() +
  geom_point() +
  labs(title = "OOB Error vs. Number of Trees in Random Forest",
       x = "Number of Trees",
       y = "Out-of-Bag Error") +
  ylim(0.15, 0.16) +
  theme_minimal()
