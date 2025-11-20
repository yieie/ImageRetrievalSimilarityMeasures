# 餘弦相似度 (Cosine Similarity) 模組分析

本模組的核心任務是實作 Cosine Similarity 演算法，並針對五種不同數據集執行全量檢索評估，以計算其平均準確率 (Avg_Precision)。

## 📂 檔案結構

根據實際目錄結構，本模組包含以下檔案與輸出資料夾：

```text
cosine/
├── cosine_with_avg_acc.py        # 核心實驗腳本 (生成所有結果的程式碼)
└── retrieval_cosine/             # 實驗結果輸出資料夾
    ├── cosine_l2_avg_acc.csv
    ├── cosine_minmax_avg_acc.csv
    ├── cosine_original_avg_acc.csv
    ├── cosine_zscore_avg_acc.csv
    └── cosine_zscore_mix_l2_avg_acc.csv

