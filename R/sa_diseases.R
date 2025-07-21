### 3 diseases VS SA components
source("./ORCI.R")

## Hypertension
# sa
hp_sa_model1 <- glm(successful_aging ~ hypertension , 
                    data = final_data, 
                    family = binomial())
summary(hp_sa_model1)
a1 <- ORCI(hp_sa_model1)

hp_sa_model2 <- glm(successful_aging ~ hypertension +sleep_duration + smoking + drinking + BMI_status,
                    data = final_data, 
                    family = binomial())

summary(hp_sa_model2)
a2 <- ORCI(hp_sa_model2)

# No major diseases
hp_md_model1 <- glm(no_major_disease ~ hypertension , 
                    data = final_data, 
                    family = binomial())
summary(hp_md_model1)
a3 <- ORCI(hp_md_model1)

hp_md_model2 <- glm(no_major_disease ~ hypertension +sleep_duration + smoking + drinking + BMI_status,
                    data = final_data, 
                    family = binomial())

summary(hp_md_model2)
a4 <- ORCI(hp_md_model2)

# No disability
hp_d_model1 <- glm(no_disability ~ hypertension , 
                   data = final_data, 
                   family = binomial())
summary(hp_d_model1)
a5 <- ORCI(hp_d_model1)

hp_d_model2 <- glm(no_disability ~ hypertension +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(hp_d_model2)
a6 <- ORCI(hp_d_model2)

# No depression
hp_yy_model1 <- glm(depressed ~ hypertension , 
                    data = final_data, 
                    family = binomial())
summary(hp_yy_model1)
a7 <- ORCI(hp_yy_model1)

hp_yy_model2 <- glm(depressed ~ hypertension +sleep_duration + smoking + drinking + BMI_status,
                    data = final_data, 
                    family = binomial())

summary(hp_yy_model2)
a8 <- ORCI(hp_yy_model2)

# High cognition
hp_hc_model1 <- glm(high_cognition ~ hypertension , 
                    data = final_data, 
                    family = binomial())
summary(hp_hc_model1)
a9 <- ORCI(hp_hc_model1)

hp_hc_model2 <- glm(high_cognition ~ hypertension +sleep_duration + smoking + drinking + BMI_status,
                    data = final_data, 
                    family = binomial())

summary(hp_hc_model2)
a10 <- ORCI(hp_hc_model2)

# active social engagement
hp_se_model1 <- glm(social_engagement ~ hypertension , 
                    data = final_data, 
                    family = binomial())
summary(hp_se_model1)
a11 <- ORCI(hp_se_model1)

hp_se_model2 <- glm(social_engagement ~ hypertension +sleep_duration + smoking + drinking + BMI_status,
                    data = final_data, 
                    family = binomial())

summary(hp_se_model2)
a12 <- ORCI(hp_se_model2)


## Kidney disease
# sa
k_sa_model1 <- glm(successful_aging ~ kidney_disease , 
                   data = final_data, 
                   family = binomial())
summary(k_sa_model1)
a13 <- ORCI(k_sa_model1)

k_sa_model2 <- glm(successful_aging ~ kidney_disease +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(k_sa_model2)
a14 <- ORCI(k_sa_model2)

# No major diseases
k_md_model1 <- glm(no_major_disease ~ kidney_disease , 
                   data = final_data, 
                   family = binomial())
summary(k_md_model1)
a15 <- ORCI(k_md_model1)

k_md_model2 <- glm(no_major_disease ~ kidney_disease +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(k_md_model2)
a16 <- ORCI(k_md_model2)

# No disability
k_d_model1 <- glm(no_disability ~ kidney_disease , 
                  data = final_data, 
                  family = binomial())
summary(k_d_model1)
a17 <- ORCI(k_d_model1)

k_d_model2 <- glm(no_disability ~ kidney_disease +sleep_duration + smoking + drinking + BMI_status,
                  data = final_data, 
                  family = binomial())

summary(k_d_model2)
a18 <- ORCI(k_d_model2)

# No depression
k_yy_model1 <- glm(depressed ~ kidney_disease , 
                   data = final_data, 
                   family = binomial())
summary(k_yy_model1)
a19 <- ORCI(k_yy_model1)

k_yy_model2 <- glm(depressed ~ kidney_disease +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(k_yy_model2)
a20 <- ORCI(k_yy_model2)

# High cognition
k_hc_model1 <- glm(high_cognition ~ kidney_disease , 
                   data = final_data, 
                   family = binomial())
summary(k_hc_model1)
a21 <- ORCI(k_hc_model1)

k_hc_model2 <- glm(high_cognition ~ kidney_disease +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(k_hc_model2)
a22 <- ORCI(k_hc_model2)

# active social engagement
k_se_model1 <- glm(social_engagement ~ kidney_disease , 
                   data = final_data, 
                   family = binomial())
summary(k_se_model1)
a23 <- ORCI(k_se_model1)

k_se_model2 <- glm(social_engagement ~ kidney_disease +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(k_se_model2)
a24 <- ORCI(k_se_model2)


## Arthritis
# sa
a_sa_model1 <- glm(successful_aging ~ arthritis, 
                   data = final_data, 
                   family = binomial())
summary(a_sa_model1)
a25 <- ORCI(a_sa_model1)

a_sa_model2 <- glm(successful_aging ~ arthritis +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(a_sa_model2)
a26 <- ORCI(a_sa_model2)

# No major diseases
a_md_model1 <- glm(no_major_disease ~ arthritis , 
                   data = final_data, 
                   family = binomial())
summary(a_md_model1)
a27 <- ORCI(a_md_model1)

a_md_model2 <- glm(no_major_disease ~ arthritis +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(a_md_model2)
a28 <- ORCI(a_md_model2)

# No disability
a_d_model1 <- glm(no_disability ~ arthritis , 
                  data = final_data, 
                  family = binomial())
summary(a_d_model1)
a29 <- ORCI(a_d_model1)

a_d_model2 <- glm(no_disability ~ arthritis +sleep_duration + smoking + drinking + BMI_status,
                  data = final_data, 
                  family = binomial())

summary(a_d_model2)
a30 <- ORCI(a_d_model2)

# No depression
a_yy_model1 <- glm(depressed ~ arthritis, 
                   data = final_data, 
                   family = binomial())
summary(a_yy_model1)
a31 <- ORCI(a_yy_model1)

a_yy_model2 <- glm(depressed ~ arthritis +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(a_yy_model2)
a32 <- ORCI(a_yy_model2)

# High cognition
a_hc_model1 <- glm(high_cognition ~ arthritis , 
                   data = final_data, 
                   family = binomial())
summary(a_hc_model1)
a33 <- ORCI(a_hc_model1)

a_hc_model2 <- glm(high_cognition ~ arthritis +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(a_hc_model2)
a34 <- ORCI(a_hc_model2)

# active social engagement
a_se_model1 <- glm(social_engagement ~ arthritis , 
                   data = final_data, 
                   family = binomial())
summary(a_se_model1)
a35 <- ORCI(a_se_model1)

a_se_model2 <- glm(social_engagement ~ arthritis +sleep_duration + smoking + drinking + BMI_status,
                   data = final_data, 
                   family = binomial())

summary(a_se_model2)
a36 <- ORCI(a_se_model2)








