#!/usr/bin/env Rscript
# ==============================================================================
# Title: A2_factor_analysis.R
# ==============================================================================
# Description: Perform exploratory and confirmatory factor analysis on behavioural data


# ==============================================================================
# Load libraries and data
# ==============================================================================
library(dplyr)
library(lavaan)
library(psych)
library(tibble)
library(ggplot2)

input_file <- file.path("..", "data", "behavioural_data_preprocessed.csv")
behavioural_df <- read.csv(input_file)

# Subtract 0-back from other RTs
behavioural_df <- behavioural_df %>%
  mutate(
    Emotion_Task_Face_Median_RT_diff = Emotion_Task_Face_Median_RT - WM_Task_0bk_Median_RT,
    Language_Task_Story_Median_RT_diff = Language_Task_Story_Median_RT - WM_Task_0bk_Median_RT,
    Social_Task_TOM_Median_RT_TOM_diff = Social_Task_TOM_Median_RT_TOM - WM_Task_0bk_Median_RT,
    ER40_CRT_diff = ER40_CRT - WM_Task_0bk_Median_RT
  )


# Split into training and test set
set.seed(42)
train_indices <- sample(1:nrow(behavioural_df), size = 0.8 * nrow(behavioural_df))
train_df <- behavioural_df[train_indices, ]
test_df <- behavioural_df[-train_indices, ]

# Scale the test set (excluding Subject and index)
scaled_train_df <- train_df %>%
  select(-X, -Subject, -Family_ID, -HasGT) %>%
  scale()

# Define variable groups
reaction_times <- c("Emotion_Task_Face_Median_RT", "Language_Task_Story_Median_RT", 
                    "Social_Task_TOM_Median_RT_TOM", "ER40_CRT")

relative_reaction_times <- c("Emotion_Task_Face_Median_RT_diff", "Language_Task_Story_Median_RT_diff", 
                             "Social_Task_TOM_Median_RT_TOM_diff", "ER40_CRT_diff")

questionnaires <- c("Friendship_Unadj", "Loneliness_Unadj", "PercHostil_Unadj", "PercReject_Unadj", 
                    "EmotSupp_Unadj", "InstruSupp_Unadj")

reaction_times_and_questionnaires <- c(reaction_times, questionnaires)
relative_reaction_times_and_questionnaires <- c(relative_reaction_times, questionnaires)

# Function to extract model fit indices, including CFI & SRMR
extract_fit <- function(efa_model, model_name) {
  return(tibble(
    Model = model_name,
    RMSEA = efa_model$RMSEA[1],
    TLI = efa_model$TLI[1],
    CFI = efa_model$CFI[1],        # Comparative Fit Index
    df = efa_model$dof,
    p_value = efa_model$PVAL,
    BIC = efa_model$BIC
  ))
}

# Run EFA on each model and extract fit statistics
efa_results <- list(
  extract_fit(fa(scaled_train_df[, reaction_times], nfactors = 1), "RT Model"),
  extract_fit(fa(scaled_train_df[, relative_reaction_times], nfactors = 1), "Relative RT Model"),
  extract_fit(fa(scaled_train_df[, questionnaires], nfactors = 1), "Questionnaire Model"),
  extract_fit(fa(scaled_train_df[, reaction_times_and_questionnaires], nfactors = 1), "RT + Questionnaire Model"),
  extract_fit(fa(scaled_train_df[, relative_reaction_times_and_questionnaires], nfactors = 1), "Relative RT + Questionnaire Model")
)

# Combine into a summary table
efa_summary_table <- bind_rows(efa_results)

# Print the summary table
print(efa_summary_table)

# Save the summary table as CSV
write.csv(efa_summary_table, file.path("..", "data", "efa_model_comparison.csv"), row.names = FALSE)

# ==============================================================================
# Test model fit with CFA in the held-out data
# ==============================================================================

scaled_test_df <- test_df %>%
  select(-X, -Subject, -Family_ID, -HasGT) %>%
  scale()

# Fitting the model with reaction times
social.model <- 'Social_Score  =~ Emotion_Task_Face_Median_RT + Language_Task_Story_Median_RT + Social_Task_TOM_Median_RT_TOM + ER40_CRT'
fit <- cfa(social.model, data = test_df)
social.model.fits <- fitMeasures(fit, fit.measures=c("rmsea", "cfi", "tli", "srmr"))
social.model.summary <- summary(fit, standardized = TRUE, rsquare = TRUE)

# Extract standardized factor loadings
standardized_loadings <- standardizedSolution(fit)
print(standardized_loadings)

# Extract R-squared values
r_squared <- inspect(fit, "rsquare")
print(r_squared)

# Saving the fit indices
fit_indices <- fitMeasures(fit, fit.measures='all')
write.csv(fit_indices, file.path("..", "data", "cfa_fit_indices.csv"), row.names = FALSE)

# Calculating the factor score for the whole sample
fit_full <- cfa(social.model, data = behavioural_df, estimator = "MLR")
factor_scores_full <- lavPredict(fit_full, type = "lv")
factor_scores_full_df <- as.data.frame(factor_scores_full)
factor_scores_full_df$Subject <- behavioural_df$Subject
factor_scores_full_df <- factor_scores_full_df %>% select(Subject, everything())
factor_scores_full_df$Social_Score <- scale(factor_scores_full_df$Social_Score) 
write.csv(
  factor_scores_full_df, 
  file=file.path("..", "data", "cfa_factor_scores_full_sample.csv"), 
  row.names = FALSE
  )

# Write output for GCA64
df <- factor_scores_full_df
colnames(df) <- c("Subject", "Social_Score")

# Add FID column (GCTA requires FID and IID, so we'll duplicate Subject)
df$IID <- behavioural_df$Subject
df$FID <- behavioural_df$Family_ID

# Reorder columns to match GCTA format (FID, IID, Phenotype)
df_gcta <- df[, c("FID", "IID", "Social_Score")]

# Write as a space-separated TXT file (without row names or column headers)
write.table(df_gcta, file.path("..", "data", "cfa_factor_scores.txt"), sep = " ", row.names = FALSE, col.names = FALSE, quote = FALSE)