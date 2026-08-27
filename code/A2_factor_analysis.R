#!/usr/bin/env Rscript

# Function to install and load required packages
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat(paste("Installing missing package:", pkg, "\n"))
      install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# List of required packages
required_packages <- c("dplyr", "lavaan", "psych", "tibble", "ggplot2")

# Install and load required packages
install_if_missing(required_packages)

# Load required libraries
library(dplyr)
library(lavaan)
library(psych)
library(tibble)
library(ggplot2)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Function to display usage
usage <- function() {
  cat("Usage: Rscript A2_factor_analysis.R --input INPUT --output OUTPUT --project PROJECT\n")
  cat("  --input     Path to preprocessed behavioural data CSV\n")
  cat("  --output    Path to save factor scores CSV\n")
  cat("  --project   Path to project directory\n")
  quit(status = 1)
}

# Parse arguments
if (length(args) == 0) {
  usage()
}

# Initialize variables
input_file <- NULL
output_file <- NULL
project_folder <- NULL

# Parse named arguments
i <- 1
while (i <= length(args)) {
  if (args[i] == "--input") {
    input_file <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--output") {
    output_file <- args[i + 1]
    i <- i + 2
  } else if (args[i] == "--project") {
    project_folder <- args[i + 1]
    i <- i + 2
  } else {
    cat("Unknown argument:", args[i], "\n")
    usage()
  }
}

# Validate required arguments
if (is.null(input_file) || is.null(output_file) || is.null(project_folder)) {
  cat("Error: Missing required arguments\n")
  usage()
}

# Create necessary directories
figures_dir <- file.path(project_folder, "figures")
reports_dir <- file.path(project_folder, "reports")
results_dir <- file.path(project_folder, "results")

if (!dir.exists(figures_dir)) {
  dir.create(figures_dir, recursive = TRUE)
}
if (!dir.exists(reports_dir)) {
  dir.create(reports_dir, recursive = TRUE)
}
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

# Initialize report
report <- c(
  paste(rep("=", 80), collapse = ""),
  "A2: FACTOR ANALYSIS REPORT",
  paste(rep("=", 80), collapse = ""),
  ""
)

# Load data
behavioural_df <- read.csv(input_file)
report <- c(report, paste("Input file:", input_file))
report <- c(report, paste("Data shape:", nrow(behavioural_df), "rows x", ncol(behavioural_df), "columns"))
report <- c(report, "")

# Subtract 0-back from other RTs
behavioural_df <- behavioural_df %>%
  mutate(
    Emotion_Task_Face_Median_RT_diff = Emotion_Task_Face_Median_RT - WM_Task_0bk_Median_RT,
    Language_Task_Story_Median_RT_diff = Language_Task_Story_Median_RT - WM_Task_0bk_Median_RT,
    Social_Task_TOM_Median_RT_TOM_diff = Social_Task_TOM_Median_RT_TOM - WM_Task_0bk_Median_RT,
    ER40_CRT_diff = ER40_CRT - WM_Task_0bk_Median_RT
  )

report <- c(report, "Created difference scores (task RT - 0-back RT)")
report <- c(report, "")

# Split into training and test set
set.seed(42)
train_indices <- sample(1:nrow(behavioural_df), size = 0.8 * nrow(behavioural_df))
train_df <- behavioural_df[train_indices, ]
test_df <- behavioural_df[-train_indices, ]

report <- c(report, "DATA SPLIT:")
report <- c(report, paste(rep("-", 80), collapse = ""))
report <- c(report, paste("Training set:", nrow(train_df), "subjects"))
report <- c(report, paste("Test set:", nrow(test_df), "subjects"))
report <- c(report, "")

# Scale the training set (excluding Subject)
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
report <- c(report, "EXPLORATORY FACTOR ANALYSIS (EFA):")
report <- c(report, paste(rep("-", 80), collapse = ""))

efa_results <- list(
  extract_fit(fa(scaled_train_df[, reaction_times], nfactors = 1), "RT Model"),
  extract_fit(fa(scaled_train_df[, relative_reaction_times], nfactors = 1), "Relative RT Model"),
  extract_fit(fa(scaled_train_df[, questionnaires], nfactors = 1), "Questionnaire Model"),
  extract_fit(fa(scaled_train_df[, reaction_times_and_questionnaires], nfactors = 1), "RT + Questionnaire Model"),
  extract_fit(fa(scaled_train_df[, relative_reaction_times_and_questionnaires], nfactors = 1), "Relative RT + Questionnaire Model")
)

# Combine into a summary table
efa_summary_table <- bind_rows(efa_results)

# Add to report
report <- c(report, capture.output(print(efa_summary_table, n = Inf)))
report <- c(report, "")

# Save the summary table as CSV
efa_comparison_file <- file.path(results_dir, "efa_model_comparison.csv")
write.csv(efa_summary_table, efa_comparison_file, row.names = FALSE)
report <- c(report, paste("EFA comparison saved to:", efa_comparison_file))
report <- c(report, "")

# ==============================================================================
# Test model fit with CFA in the held-out data
# ==============================================================================

report <- c(report, "CONFIRMATORY FACTOR ANALYSIS (CFA) - TEST SET:")
report <- c(report, paste(rep("-", 80), collapse = ""))

scaled_test_df <- test_df %>%
  select(-X, -Subject, -Family_ID, -HasGT) %>%
  scale()

# Fitting the model with reaction times
social.model <- 'Social_Score  =~ Emotion_Task_Face_Median_RT + Language_Task_Story_Median_RT + Social_Task_TOM_Median_RT_TOM + ER40_CRT'
fit <- cfa(social.model, data = test_df)
social.model.fits <- fitMeasures(fit, fit.measures=c("rmsea", "cfi", "tli", "srmr"))

report <- c(report, "Model specification:")
report <- c(report, social.model)
report <- c(report, "")

report <- c(report, "Fit measures:")
report <- c(report, capture.output(print(social.model.fits)))
report <- c(report, "")

# Extract standardized factor loadings
standardized_loadings <- standardizedSolution(fit)
report <- c(report, "Standardized factor loadings:")
report <- c(report, capture.output(print(standardized_loadings)))
report <- c(report, "")

# Extract R-squared values
r_squared <- inspect(fit, "rsquare")
report <- c(report, "R-squared values:")
report <- c(report, capture.output(print(r_squared)))
report <- c(report, "")

# Saving the fit indices
fit_indices <- fitMeasures(fit, fit.measures='all')
fit_indices_file <- file.path(results_dir, "cfa_fit_indices.csv")
write.csv(fit_indices, fit_indices_file, row.names = FALSE)
report <- c(report, paste("Fit indices saved to:", fit_indices_file))
report <- c(report, "")

# ==============================================================================
# Calculate factor scores for full sample
# ==============================================================================

report <- c(report, "FACTOR SCORES - FULL SAMPLE:")
report <- c(report, paste(rep("-", 80), collapse = ""))

# Calculating the factor score for the whole sample
fit_full <- cfa(social.model, data = behavioural_df, estimator = "MLR")
factor_scores_full <- lavPredict(fit_full, type = "lv")
factor_scores_full_df <- as.data.frame(factor_scores_full)
factor_scores_full_df$Subject <- behavioural_df$Subject
factor_scores_full_df <- factor_scores_full_df %>% select(Subject, everything())
factor_scores_full_df$Social_Score <- scale(factor_scores_full_df$Social_Score)

report <- c(report, paste("Calculated factor scores for", nrow(factor_scores_full_df), "subjects"))
report <- c(report, "")

# Descriptive statistics of factor scores
report <- c(report, "Descriptive statistics of Social_Score:")
score_summary <- summary(factor_scores_full_df$Social_Score)
report <- c(report, capture.output(print(score_summary)))
report <- c(report, "")

# Save factor scores (main output)
write.csv(
  factor_scores_full_df,
  file = output_file,
  row.names = FALSE
)

report <- c(report, paste("Factor scores saved to:", output_file))
report <- c(report, "")

# Write output for GCTA64
df <- factor_scores_full_df
colnames(df) <- c("Subject", "Social_Score")

# Add FID column (GCTA requires FID and IID, so we'll duplicate Subject)
df$IID <- behavioural_df$Subject
df$FID <- behavioural_df$Family_ID

# Reorder columns to match GCTA format (FID, IID, Phenotype)
df_gcta <- df[, c("FID", "IID", "Social_Score")]

# Write as a space-separated TXT file (without row names or column headers)
gcta_file <- file.path(results_dir, "cfa_factor_scores.txt")
write.table(df_gcta, gcta_file, sep = " ", row.names = FALSE, col.names = FALSE, quote = FALSE)

report <- c(report, paste("GCTA-format file saved to:", gcta_file))
report <- c(report, "")

# ==============================================================================
# Create visualization of factor loadings
# ==============================================================================

report <- c(report, "VISUALIZATIONS:")
report <- c(report, paste(rep("-", 80), collapse = ""))

# Create a plot of standardized loadings
loadings_plot_data <- standardized_loadings %>%
  filter(op == "=~") %>%
  select(rhs, est.std) %>%
  rename(Variable = rhs, Loading = est.std)

png(file.path(figures_dir, "A2_factor_loadings.png"), width = 800, height = 600, res = 120)
p <- ggplot(loadings_plot_data, aes(x = reorder(Variable, Loading), y = Loading)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Standardized Factor Loadings",
    x = "Variable",
    y = "Standardized Loading"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  )
print(p)
dev.off()

report <- c(report, paste("Factor loadings plot saved to:", file.path(figures_dir, "A2_factor_loadings.png")))

# Create histogram of factor scores
png(file.path(figures_dir, "A2_factor_scores_distribution.png"), width = 800, height = 600, res = 120)
p2 <- ggplot(factor_scores_full_df, aes(x = Social_Score)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(
    title = "Distribution of Social Factor Scores",
    x = "Social Score (standardized)",
    y = "Count"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12)
  )
print(p2)
dev.off()

report <- c(report, paste("Factor scores distribution plot saved to:", file.path(figures_dir, "A2_factor_scores_distribution.png")))
report <- c(report, "")

# ==============================================================================
# Write report to file
# ==============================================================================

report <- c(report, paste(rep("=", 80), collapse = ""))
report <- c(report, "END OF REPORT")
report <- c(report, paste(rep("=", 80), collapse = ""))

report_file <- file.path(reports_dir, "A2_factor_analysis_report.txt")
writeLines(report, report_file)

cat("Analysis complete. Report saved to:", report_file, "\n")
