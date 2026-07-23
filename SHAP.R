# =======================================================
# 0. 环境设置与包加载
# =======================================================
# 定义需要的包列表
required_packages <- c("tidyverse", "caret", "lightgbm", "shapviz", "ggplot2", "doParallel")

# 自动安装缺失的包
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
  print(paste("Installing missing packages:", paste(new_packages, collapse = ", ")))
  install.packages(new_packages)
}

# 加载包
library(tidyverse)
library(caret)
library(lightgbm)    # 核心模型
library(shapviz)     # SHAP 解释
library(ggplot2)     # 绘图
library(doParallel)  # 并行计算

# 启用并行计算 (加速数据处理)
detected_cores <- detectCores()
worker_count <- if (is.na(detected_cores)) 1 else max(1, detected_cores - 1)
cl <- NULL
if (worker_count > 1) {
  cl <- makePSOCKcluster(worker_count)
  registerDoParallel(cl)
  print(paste("✅ Environment ready.", worker_count, "parallel workers enabled."))
} else {
  registerDoSEQ()
  print("✅ Environment ready. Sequential processing enabled.")
}

# =======================================================
# 1. 数据读取与预处理
# =======================================================
# 请确保路径正确
data_path <- "alzheimers_disease_data.csv"

if(!file.exists(data_path)) stop("Error: File not found. Check the path!")
data <- read.csv(data_path)

# 剔除无关ID和泄露特征 (FunctionalAssessment)
# 保留 MMSE, ADL, MemoryComplaints 等关键特征
cols_to_remove <- c("PatientID", "DoctorInCharge", "FunctionalAssessment")
cols_to_remove <- cols_to_remove[cols_to_remove %in% names(data)]
data_clean <- data %>% select(-all_of(cols_to_remove))

print(paste("🧹 Data Cleanup: Removed columns:", paste(cols_to_remove, collapse=", ")))

if ("SystolicBP" %in% names(data_clean) && "DiastolicBP" %in% names(data_clean)) {
  data_clean$PulsePressure <- data_clean$SystolicBP - data_clean$DiastolicBP
}

# =======================================================
# 2. 准备 LightGBM 数据格式
# =======================================================
# 处理特征矩阵
# LightGBM 需要数值矩阵。将所有 Factor 转为 Numeric
data_task <- data_clean %>%
  select(-Diagnosis) %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), as.numeric))

data_matrix <- as.matrix(data_task)

# 填补可能的 NA
if(any(is.na(data_matrix))) {
  data_matrix[is.na(data_matrix)] <- 0
  print("⚠️ Warning: Found NAs in matrix, filled with 0")
}

# =======================================================
# 3. 加载主训练流程导出的 LightGBM 模型
# =======================================================
model_path <- "final_model_lgb.txt"
if (!file.exists(model_path)) {
  stop("Error: LightGBM model not found. Run ML_design.R first.")
}
lgb_model <- lgb.load(model_path)
print("✅ Exported LightGBM model loaded.")

# =======================================================
# 4. 计算 SHAP 值
# =======================================================
print("🔍 Calculating SHAP values...")

# shapviz 能够自动识别 LightGBM 模型并计算 SHAP
# 注意：LightGBM 的 SHAP 默认基于 Log-odds (Raw Score)
sv <- shapviz(lgb_model, X_pred = data_matrix)

# =======================================================
# 5. 绘制并保存高清图 (修正标签版)
# =======================================================

# 设置文件名
output_filename <- "SHAP_Summary_LightGBM_LogOdds.png"

# 打开绘图设备
png(output_filename, width = 10, height = 6, units = "in", res = 300)

# 绘图逻辑
p <- sv_importance(sv, kind = "beeswarm", show_numbers = TRUE, max_display = 20) +
  scale_color_gradient(low = "#0073C2FF", high = "#EF553BFF") +
  theme_bw() +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, color = "gray30"),
    axis.text.y = element_text(color = "black", size = 11, face = "bold"),
    legend.position = "right",
    legend.title = element_text(face = "bold")
  ) +
  labs(
    title = "Feature Importance: LightGBM (SHAP)",
    # 这里的 subtitle 解释了为什么数值看起来比 RF 大
    subtitle = "Top Predictors based on Log-odds contribution (Raw Output)",
    # X 轴明确单位
    x = "SHAP Value (Log-odds Impact: >0 increases Risk)",
    color = "Feature Value"
  )

print(p)
dev.off() # 关闭设备，保存文件

print(paste("🎉 Plot saved successfully as:", output_filename))

if (!is.null(cl)) stopCluster(cl)
registerDoSEQ()
