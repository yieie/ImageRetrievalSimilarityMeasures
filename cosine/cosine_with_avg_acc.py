import numpy as np
import pandas as pd
import os
import time

# --- A. Cosine Similarity 函式 (您的核心實作) ---
def cosine_similarity(x, y):
    """
    計算兩個 NumPy 向量 x 和 y 之間的餘弦相似度 (值越大越相似)。
    數學公式: cos(theta) = (A · B) / (||A|| * ||B||)
    """
    
    # 1. 計算分子 (內積 A · B)
    dot_product = np.dot(x, y) 
    
    # 2. 計算分母 (||A|| * ||B||)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    denominator = norm_x * norm_y
    
    # 3. 檢查分母是否為零 (避免除以零錯誤，零向量相似度設為 0)
    if denominator == 0:
        return 0.0
    
    # 4. 計算最終相似度
    similarity = dot_product / denominator
    
    return similarity

# --- B. 檢索與評估主體函式 ---
def evaluate_retrieval(X_data, Y_labels, k=10):
    """
    執行圖像檢索並計算平均準確率 (Precision)。
    
    Args:
        X_data (np.array): 特徵矩陣 (10000 x 478)。
        Y_labels (np.array): 圖片類別標籤 (10000)。
        k (int): 檢索回傳的數量 (Top-10)。
        
    Returns:
        float: 所有查詢的平均準確率。
    """
    N = X_data.shape[0]
    total_precision = 0.0
    
    # 遍歷所有 N 張圖作為 Query
    for i in range(N):
        query_vector = X_data[i]
        query_label = Y_labels[i]
        
        similarity_scores = []
        
        # 計算 Query 與所有其他圖片的相似度
        for j in range(N):
            if i == j:
                # 排除 Query 自己
                continue 
            
            target_vector = X_data[j]
            # 呼叫 Cosine Similarity 函式
            score = cosine_similarity(query_vector, target_vector) 
            similarity_scores.append((score, j))

        # 排序：Cosine Similarity 是相似度指標，越大越相似，因此使用降序 (reverse=True)
        # 由於 N=10000，排序可能較慢，但這是獲取 Top-10 必須的步驟。
        similarity_scores.sort(key=lambda item: item[0], reverse=True)
        
        top_k_results = similarity_scores[:k]
        
        correct_count = 0
        
        # 計算準確率 (Precision)
        for score, index in top_k_results:
            retrieved_label = Y_labels[index]
            if retrieved_label == query_label:
                # 答對的數量
                correct_count += 1
        
        # Precision = |Correct| / |Retrieved| = correct_count / k
        precision = correct_count / k 
        total_precision += precision
        
    # 計算所有查詢的平均準確率 (Mean Average Precision)
    avg_precision = total_precision / N
    return avg_precision

# --- C. 數據讀取與實驗執行主體 ---

# 定義要測試的五種數據集檔案名稱 (對應 Normalization.py 的輸出)
data_files = [
    'ImageRetrievalSimilarityMeasures/fullset_original.txt',
    'ImageRetrievalSimilarityMeasures/fullset_zscore.txt',
    'ImageRetrievalSimilarityMeasures/fullset_l2.txt',
    'ImageRetrievalSimilarityMeasures/fullset_minmax.txt',
    'ImageRetrievalSimilarityMeasures/fullset_zscore_mix_l2.txt'
]

output_dir = 'retrieval_cosine'
os.makedirs(output_dir, exist_ok=True) # 創建輸出資料夾
results = []

print("--- 開始執行 Cosine Similarity 檢索實驗 ---")

for file_name in data_files:
    # 假設檔案與此腳本位於同一或上一層目錄
    file_path = os.path.join('.', file_name) 
    
    # -----------------------------------------------------------------
    # 核心修正點：
    # 1. 從 'ImageRetrievalSimilarityMeasures/fullset_original.txt' 中提取檔案名
    base_name = os.path.basename(file_path) # 這會得到 'fullset_original.txt'
    
    # 2. 從檔案名稱解析數據類型
    data_type = base_name.replace('fullset_', '').replace('.txt', '') # 這會得到 'original'
    # -----------------------------------------------------------------

    print(f"\n>> 正在處理檔案: {file_name}")
    
    try:
        start_time = time.time()
        
        # 讀取數據：使用 Pandas 讀取所有欄位
        df = pd.read_csv(file_path, header=None, sep='\s+')
        
        # -----------------------------------------------------------------
        # 核心修正點：
        # 根據 README，無用的 80-335 欄已被移除。
        # 實際特徵維度是 222，而不是 478。
        # -----------------------------------------------------------------
        FEATURE_DIM = 222
        
        # 1. 讀取數據：使用正則表達式 r'\s+' 處理任意數量的空格
        df = pd.read_csv(
            file_path, 
            header=None, 
            sep=r'\s+',        # 彈性分隔符
            engine='python',   
            skiprows=1,        # 跳過標頭行
            skipinitialspace=True 
        )
        
        # 檢查讀取到的總欄位數 (222 特徵 + 1 檔名 + 1 標籤 = 224 欄)
        if df.shape[1] < FEATURE_DIM + 2:
            raise Exception(f"數據讀取失敗：預期至少 {FEATURE_DIM + 2} 欄，實際只有 {df.shape[1]} 欄。")

        # 2. 分離特徵 (X)
        # 僅選取前 222 欄 (0 到 221)
        X = df.iloc[:, :FEATURE_DIM].values.astype(float)
        
        # 3. 分離標籤 (Y)
        # 標籤是最後一欄
        Y = df.iloc[:, -1].values 
        
        # -----------------------------------------------------------------


        print(f"數據載入成功。特徵形狀: {X.shape}, 標籤數量: {Y.shape[0]}")
        
        # 執行檢索評估
        avg_prec = evaluate_retrieval(X, Y)
        
        end_time = time.time()
        
        # 儲存結果到 CSV
        output_file = os.path.join(output_dir, f'cosine_{data_type}_avg_acc.csv')
        
        result_df = pd.DataFrame([{
            'DataSet': file_name, 
            'Method': 'Cosine', 
            'Avg_Precision': avg_prec
        }])
        result_df.to_csv(output_file, index=False)
        
        print(f"**結果已儲存** 到 {output_file}")
        print(f"平均準確率 ({data_type}): {avg_prec:.4f}")
        print(f"耗時: {end_time - start_time:.2f} 秒")
        
        results.append({
            'DataSet': data_type,
            'Avg_Precision': avg_prec
        })
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}。請確保前置步驟 (Mix_Dataset.py, Normalization.py) 已正確運行，且檔案位於正確路徑。")
    except Exception as e:
        print(f"處理檔案 {file_path} 時發生錯誤: {e}")

print("\n--- Cosine Similarity 實驗執行完畢 ---")