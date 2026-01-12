import re
import glob
import os
from collections import Counter

def analyze_rewards_per_file(path_pattern):
    """
    支援萬用字元，並針對每個符合的檔案進行獨立統計
    """
    # 匹配 "Accumulated_reward: " 後面的數字
    pattern = re.compile(r"Accumulated_reward:\s*(\d+)")
    
    # 找出所有符合模式的檔案
    file_list = glob.glob(path_pattern)
    
    if not file_list:
        print(f"❌ 找不到符合模式的檔案: {path_pattern}")
        return

    print(f"📂 找到 {len(file_list)} 個檔案，開始個別統計...\n")

    for file_path in sorted(file_list):
        if os.path.isfile(file_path):
            file_rewards = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        match = pattern.search(line)
                        if match:
                            file_rewards.append(int(match.group(1)))
                
                # 針對目前的檔案進行統計
                counts = Counter(file_rewards)
                
                # 印出該檔案的結果
                print(f"📄 檔案名稱: {os.path.basename(file_path)}")
                if not file_rewards:
                    print("   (此檔案中未發現統計數據)")
                else:
                    print(f"   {'Reward':<15} | {'Step Count':<10}")
                    print("   " + "-" * 28)
                    for reward in sorted(counts.keys()):
                        print(f"   {reward:<15} | {counts[reward]:<10}")
                
                print("\n" + "="*40 + "\n")
                
            except Exception as e:
                print(f"❌ 讀取檔案 {file_path} 時發生錯誤: {e}\n")

# --- 使用範例 ---
if __name__ == "__main__":
    # 您可以根據需求修改路徑模式
    # 例如：'logs/*.txt' 或 'test_result_*.log'
    analyze_rewards_per_file('log/game3-1_MC-DML_revised*')