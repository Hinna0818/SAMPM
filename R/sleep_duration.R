## sleep duration vs SA
# using RCS and logistics regression to construct the model

library(rms)
library(jstable)
library(ggplot2)
source("./ORCI.R")

## RCS
dd <- datadist(final_data)
options(datadist = "dd")

sleep_sa <- lrm(successful_aging ~ rcs(sleep_duration, 3) + smoking + drinking + BMI_status, 
                data = final_data)

## non-linear testing
anova(sleep_sa)

OR_sleep <- Predict(sleep_sa, sleep_duration, fun=exp)

## rcs plot
ggplot() +
  geom_histogram(data = final_data,
                 aes(x = sleep_duration, y = ..density.. * 12, fill = "Sleep duration Distribution", ),
                 binwidth = 1,
                 color = "white", alpha = 0.5) +
  geom_line(data = OR_sleep,
            aes(x = sleep_duration, y = yhat, color = "OR (95% CI)"), size = 1) +
  geom_ribbon(data = OR_sleep,
              aes(x = sleep_duration, ymin = lower, ymax = upper),
              fill = "#FF4500", alpha = 0.1) +
  geom_hline(yintercept = 1, linetype = 2, size = 0.6) +
  scale_y_continuous(
    name = "OR (95% CI)",
    sec.axis = sec_axis(~ . / 1.5, name = "Frequency density")
  ) +
  scale_fill_manual(
    name = "",
    values = ("Sleep duration Distribution" = "grey")
  ) +
  scale_color_manual(
    name = "",
    values = c("OR (95% CI)" = "#FF4500")
  ) +
  labs(title = "Successful Aging",
       x = "Sleep duration") +
  theme_classic() +
  theme(
    axis.title.y.left = element_text(color = "black", size = 12),
    axis.title.y.right = element_text(color = "black", size = 12),
    axis.line.y.left = element_line(color = "black"),
    axis.line.y.right = element_line(color = "black"),
    plot.title = element_text(size = 16),
    legend.position = c(0.01, 1.05),
    legend.justification = c("left", "top"),
    legend.direction = "vertical",    
    legend.box = "vertical",
    legend.spacing.y = unit(0.5, "pt")) +
  coord_cartesian(xlim = c(2, 12)) +   
  theme(aspect.ratio = 0.9)

final_data$sleep_group_update <- cut(final_data$sleep_duration,
                                     breaks = c(-Inf, 4, 6, 8, 10, Inf),
                                     labels = c("<4 h", "4–6 h", "6–8 h", "8–10 h", "≥10 h"),
                                     right = FALSE)  # 左闭右开，4小时属于4–6 h组

## logistic regression 
final_data$sleep_group_update <- relevel(final_data$sleep_group_update, ref = "6–8 h")

## model0
fit_group_update0 <- glm(successful_aging ~ sleep_group_update, 
                         family = binomial, data = final_data)
## model1
fit_group_update <- glm(successful_aging ~ sleep_group_update + smoking + drinking + BMI_status, 
                        family = binomial, data = final_data)

summary(fit_group_update0)
summary(fit_group_update)

sd0 <- ORCI(fit_group_update0)
sd1 <- ORCI(fit_group_update)

sd0
sd1
