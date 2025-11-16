import numpy as np




__SEARCH_ITSELF__ = False
def load_data(filename):
    data = []
    names = []
    classes = []   # 每 200 row 為一類別

    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()

            img_name = parts[-2]
            img_class = i // 200   # ★每 200 row 一類★

            features = np.array(list(map(float, parts[:-2])))

            data.append(features)
            names.append(img_name)
            classes.append(img_class)

    return np.array(data), names, classes


def pcc_similarity(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0
    return np.corrcoef(a, b)[0, 1]


def search_top10(query_row, data, names, classes):
    if query_row < 0 or query_row >= len(data):
        print("row 超出範圍")
        return

    query_vec = data[query_row]
    query_class = classes[query_row]

    scores = []

    for i in range(len(data)):
        if i == query_row and not __SEARCH_ITSELF__:
            continue
        sim = pcc_similarity(query_vec, data[i])
        scores.append((i, classes[i], sim))   # ★使用 row index★

    # 依 PCC 倒序
    scores.sort(key=lambda x: x[2], reverse=True)

    top10 = scores[:10]

    print("\n=== PCC Top-10 ===")
    correct = 0

    for row_id, cls, sim in top10:
        if cls == query_class:
            correct += 1
        print(f"{row_id:<6d}    PCC={sim:.6f}")

    accuracy = correct / 10
    print(f"accuracy： {accuracy:.2f}")


# ================ 主程式 ================
if __name__ == "__main__":
    filename = "fullset_minmax.txt"
    data, names, classes = load_data(filename)

    query_row = int(input("請輸入要查詢的 row（從0開始）："))
    search_top10(query_row, data, names, classes)
