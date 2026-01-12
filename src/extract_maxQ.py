import re
import glob
import os

def extract_max_q_with_wildcards(path_pattern):
    """
    支援萬用字元，讀取多個遊戲日誌檔，
    提取每個 Step 的最大 Q 值並輸出至 stdout。
    """
    # 取得所有符合條件的檔案並排序
    file_list = sorted(glob.glob(path_pattern))
    
    if not file_list:
        print(f"❌ 找不到符合模式的檔案: {path_pattern}")
        return

    # 正則表達式：匹配 Q 值（整數或浮點數）
    q_pattern = re.compile(r'Q=([-+]?\d*\.\d+|\d+)')
    # 正則表達式：切割 Step 區塊
    step_splitter = re.compile(r'[Ss]tep:\s+')

    for file_path in file_list:
        if not os.path.isfile(file_path):
            continue

        print(f"📄 檔案分析: {os.path.basename(file_path)}")
        print("-" * 40)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 切割出各個 Step 區塊
            steps = step_splitter.split(content)
            
            found_data = False
            for step_content in steps:
                if not step_content.strip():
                    continue
                
                # 第一行通常是 Step ID
                lines = step_content.split('\n')
                step_id = lines[0].strip()
                
                # 尋找該 Step 內所有的 Q 值
                q_values = q_pattern.findall(step_content)
                
                if q_values:
                    # 轉為 float 並取最大值
                    max_q = max(float(q) for q in q_values)
                    print(f"  Step {step_id}: Max Q = {max_q}")
                    found_data = True
                else:
                    # 如果該 Step 裡沒有 Q 值（可選是否顯示）
                    # print(f"  Step {step_id}: No Q values found")
                    pass
            
            if not found_data:
                print("  (此檔案中未發現任何 Q 值數據)")

        except Exception as e:
            print(f"  ❌ 處理檔案時發生錯誤: {e}")
        
        print("\n" + "="*40 + "\n")

# --- 使用範例 ---
if __name__ == "__main__":
    # 您可以傳入萬用字元路徑，例如 'log/game3-1_*' 或 'log/*.txt'
    # 這裡以您之前的路徑格式為範例
    extract_max_q_with_wildcards('log/game1-2_MCTS_*')