# Universal-Stat-Analysis

Limbitless Universal Statistical Analysis Script

# NORMALIZATION SCRIPT

- Load in desired .xslx or .csv sheet into ‘excel_sheets’ folder

- Normalize data by running the command in terminal: python normalize.py excel_sheets/insertname.xlsx (or .csv)

- This loads a new .csv file inside the excel_sheets file which will be read by the stats_runner.py script to perform statistical analysis

- Run desired test in terminal based on the command references below inside stat_runner.py script

# STAT RUNNER SCRIPT

---------- Independent t-test ----------

# FULL COHORT (Cohort 1 vs 2)
python stats_runner.py excel_sheets/insertname_tidy.csv --ttest_ind --value value --group Cohort --levels 1,2

# COHORT 1 vs 2 (Females only)
python stats_runner.py excel_sheets/insertname_tidy.csv --ttest_ind --value value --group Cohort --levels 1,2 --filter Sex=F

# EQUAL VARIANCES (Student t-test)
python stats_runner.py excel_sheets/insertname_tidy.csv --ttest_ind --equal_var --value value --group Cohort --levels 1,2

# AGE RANGE (18–20)
python stats_runner.py excel_sheets/insertname_tidy.csv --ttest_ind --value value --group Cohort --levels 1,2 --filter "Age>=18,Age<=20"



---------- Paired t-test ----------

# R vs L (needs SubjectID)
python stats_runner.py excel_sheets/insertname_tidy.csv --ttest_rel --value value --subject SubjectID --condition side --levels R,L

# R vs L (Cohort 1 only)
python stats_runner.py excel_sheets/insertname_tidy.csv --ttest_rel --value value --subject SubjectID --condition side --levels R,L --filter Cohort=1


 ---------- One-way ANOVA ----------

# Across trial_id (all cohorts)
python stats_runner.py excel_sheets/insertname_tidy.csv --anova1 --value value --factor trial_id

# Across trial_id (Females only)
python stats_runner.py excel_sheets/insertname_tidy.csv --anova1 --value value --factor trial_id --filter Sex=F

# Across trial_id (Age 18–20)
python stats_runner.py excel_sheets/insertname_tidy.csv --anova1 --value value --factor trial_id --filter "Age>=18,Age<=20"



---------- Two-way ANOVA ----------

# Cohort × side
python stats_runner.py excel_sheets/insertname_tidy.csv --anova2 --value value --factorA Cohort --factorB side

# Cohort × side (Females only)
python stats_runner.py excel_sheets/insertname_tidy.csv --anova2 --value value --factorA Cohort --factorB side --filter Sex=F



---------- Correlations ----------

# Pearson (FULL COHORT)
python stats_runner.py excel_sheets/insertname_tidy.csv --pearson --x Sleep_hours --y value

# Spearman (COHORT 2 only)
python stats_runner.py excel_sheets/insertname_tidy.csv --spearman --x Screen_time --y value --filter Cohort=2

# Kendall (BY GENDER: Females)
python stats_runner.py excel_sheets/insertname_tidy.csv --kendall --x Age --y value --filter Sex=F

# Pearson (BY AGE 18–20)
python stats_runner.py excel_sheets/insertname_tidy.csv --pearson --x Sleep_hours --y value --filter "Age>=18,Age<=20"



 ---------- Tukey HSD ----------

# Pairwise comparisons across trial_id
python stats_runner.py excel_sheets/insertname_tidy.csv --tukey --value value --group trial_id

# Tukey across trial_id (Cohort 1 only)
python stats_runner.py excel_sheets/insertname_tidy.csv --tukey --value value --group trial_id --filter Cohort=1



 ---------- FDR Adjustment ----------

# Adjust p-values in a separate CSV
python stats_runner.py excel_sheets/pvals.csv --fdr --pcol pval --alpha 0.05


---------- Bonus combos ----------

# Pearson (Cohort 1, Females, Age 18–20)
python stats_runner.py excel_sheets/insertname_tidy.csv --pearson --x Sleep_hours --y value --filter "Cohort=1,Sex=F,Age>=18,Age<=20"

# Two-way ANOVA (Cohort × side, Age > 18)
python stats_runner.py excel_sheets/insertname_tidy.csv --anova2 --value value --factorA Cohort --factorB side --filter "Age>18"








