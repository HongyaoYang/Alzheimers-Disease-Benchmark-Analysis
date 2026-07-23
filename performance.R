library(tidyverse)
library(scales)

# ==============================================================================
# 1. 准备数据
# ==============================================================================
csv_data <- '
"Model","AUC","PRC","Sensitivity","Specificity","Accuracy"
"LightGBM",0.873028690860726,0.810975817507315,0.870036101083033,0.69078947368421,0.806526806526807
"NaiveBayes",0.864953448603458,0.798730181399296,1,0,0.645687645687646
"RandomForest",0.867922287668631,0.796337834667288,0.873646209386282,0.651315789473684,0.794871794871795
"ElasticNet",0.82493349800494,0.744661600065024,0.898916967509025,0.480263157894737,0.750582750582751
"Logistic",0.822297168915067,0.748382490316838,0.833935018050541,0.572368421052632,0.741258741258741
"SVM",0.813343150294509,0.740808791054013,0.848375451263538,0.572368421052632,0.750582750582751
"kNN",0.682061086832605,0.534765922935195,0.873646209386282,0.322368421052632,0.678321678321678
"TabNet(DL)",0.9012684589170769,0.8656889504224513,0.631578947368421,0.9532374100719424,0.8395348837209302

'

df <- read.csv(text = csv_data, stringsAsFactors = FALSE)

# ==============================================================================
# 2. 数据处理
# ==============================================================================
df_long <- df %>%
  # 转换为长格式
  pivot_longer(cols = c("AUC", "PRC", "Sensitivity", "Specificity", "Accuracy"),
               names_to = "Metric",
               values_to = "Score") %>%
  # 1. 过滤掉 Sensitivity
  filter(Metric != "Sensitivity")

# 2. 设置模型排序 (按 AUC 最高排在最上面)
df_long <- df_long %>%
  mutate(Model = fct_reorder(Model, Score, .fun = max, .desc = FALSE))

# 3. 【关键】设置指标的显示顺序 (AUC -> PRC -> Accuracy -> Specificity)
df_long$Metric <- factor(df_long$Metric,
                         levels = c("AUC", "PRC", "Accuracy", "Specificity"))

# ==============================================================================
# 3. 绘图
# ==============================================================================
p <- ggplot(df_long, aes(x = Model, y = Score, fill = Metric)) +
  # 柱状图
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +

  # 数值标签
  geom_text(aes(label = sprintf("%.2f", Score)),
            position = position_dodge(width = 0.8),
            hjust = -0.15,
            size = 3,
            color = "black",
            fontface = "bold") +

  # 坐标轴翻转
  coord_flip(clip = "off") +

  # Y轴范围设置
  scale_y_continuous(limits = c(0, 1.15), expand = c(0, 0)) +

  # 【配色方案：稳重、专业、区分度高】
  # 选用了一组经典的扁平化设计颜色 (Flat Design)
  scale_fill_manual(values = c(
    "AUC"         = "#C0392B",  # 深红色 (权威/重点)
    "PRC"         = "#2980B9",  # 深蓝色 (沉稳)
    "Accuracy"    = "#27AE60",  # 深绿色 (通过/正确)
    "Specificity" = "#8E44AD"   # 深紫色 (独特/区分)
  )) +

  # 主题设置
  theme_minimal(base_size = 14) +
  labs(title = "Model Performance on Independent Test Set",

       x = NULL, y = "Score", fill = "Metric") +

  theme(
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(face = "bold", color = "black", size = 11),
    legend.box = "horizontal",
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
    plot.margin = margin(10, 20, 10, 10)
  )

print(p)
ggsave("matric_compare.png", plot = p, width = 10, height = 7, dpi = 300)
