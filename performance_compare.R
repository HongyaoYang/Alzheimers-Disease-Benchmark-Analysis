# === 1. 加载必要的库 ===
library(pROC)     # 用于 ROC
library(PRROC)    # 用于 PRC (关键库)
library(ggplot2)
library(dplyr)
library(gridExtra) # 用于把两张图拼在一起 (可选)

# === 2. 准备预测概率 ===
# 确保真实标签是数值型 (0/1) 用于 PRC 计算
# 假设 test_set$Diagnosis 是 Factor "No"/"Yes"
truth_numeric <- ifelse(test_set$Diagnosis == "Yes", 1, 0)

# 获取 LightGBM 概率
prob_lgb <- predict(model_lgb, test_matrix)

# 获取 Elastic Net 概率
prob_enet <- predict(model_enet, test_set, type = "prob")$Yes

# === 3. 生成 ROC 数据与绘图 ===
# 创建 roc 对象
roc_lgb  <- roc(test_set$Diagnosis, prob_lgb, quiet = TRUE)
roc_enet <- roc(test_set$Diagnosis, prob_enet, quiet = TRUE)

# 创建列表
roc_list <- list(
  "LightGBM" = roc_lgb,
  "ElasticNet" = roc_enet
)

# 动态生成带 AUC 的标签
label_lgb_roc <- paste0("LightGBM (AUC = ", sprintf("%.3f", auc(roc_lgb)), ")")
label_enet_roc <- paste0("ElasticNet (AUC = ", sprintf("%.3f", auc(roc_enet)), ")")

# 绘图
p_roc <- ggroc(roc_list, legacy.axes = TRUE, size = 1.2) +
  coord_equal() +
  scale_color_manual(
    values = c("LightGBM" = "#E41A1C", "ElasticNet" = "#377EB8"),
    labels = c(label_lgb_roc, label_enet_roc)
  ) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  theme_bw(base_size = 14) +
  labs(
    title = "ROC Curve Comparison between LightGBM and ElasticNet",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = "Model"
  ) +
  theme(
    legend.position = c(0.75, 0.15), # 图例位置放在右下
    legend.background = element_rect(fill = alpha("white", 0.8), color = "black", linewidth = 0.3),
    plot.title = element_text(face = "bold", size = 16),
    panel.grid.minor = element_blank()
  )

print(p_roc)


# === 4. 生成 PRC 数据与绘图 ===

# 定义一个辅助函数来提取 PRC 绘图数据
get_pr_data <- function(probs, truth, model_name) {
  pr_obj <- pr.curve(scores.class0 = probs, weights.class0 = truth, curve = TRUE)
  # pr_obj$curve 包含三列: Recall, Precision, Threshold
  df <- as.data.frame(pr_obj$curve)
  colnames(df) <- c("Recall", "Precision", "Threshold")
  df$Model <- model_name
  df$AUC_PR <- pr_obj$auc.integral # 保存 AUC-PR 值
  return(df)
}

# 提取两个模型的数据
pr_data_lgb <- get_pr_data(prob_lgb, truth_numeric, "LightGBM")
pr_data_enet <- get_pr_data(prob_enet, truth_numeric, "ElasticNet")

# 合并数据
pr_data_final <- rbind(pr_data_lgb, pr_data_enet)

# 动态生成带 AUPRC 的标签
auc_pr_lgb <- unique(pr_data_lgb$AUC_PR)
auc_pr_enet <- unique(pr_data_enet$AUC_PR)

label_lgb_pr <- paste0("LightGBM (AUC = ", sprintf("%.3f", auc_pr_lgb), ")")
label_enet_pr <- paste0("ElasticNet (AUC = ", sprintf("%.3f", auc_pr_enet), ")")

# 绘图
p_prc <- ggplot(pr_data_final, aes(x = Recall, y = Precision, color = Model)) +
  geom_line(size = 1.2) +
  coord_equal() + # 保持比例
  scale_color_manual(
    values = c("LightGBM" = "#E41A1C", "ElasticNet" = "#377EB8"),
    labels = c("LightGBM" = label_lgb_pr, "ElasticNet" = label_enet_pr)
  ) +
  # 添加基准线 (Baseline) = 正样本比例
  geom_hline(yintercept = sum(truth_numeric)/length(truth_numeric),
             linetype = "dashed", color = "grey50") +

  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +

  theme_bw(base_size = 14) +
  labs(
    title = "PRC Curve Comparison between LightGBM and ElasticNet",
    x = "Recall (Sensitivity)",
    y = "Precision",
    color = "Model"
  ) +
  theme(
    legend.position = c(0.19, 0.15), # 图例位置放在左下
    legend.background = element_rect(fill = alpha("white", 0.8), color = "black", linewidth = 0.3),
    plot.title = element_text(face = "bold", size = 16),
    panel.grid.minor = element_blank()
  )

print(p_prc)