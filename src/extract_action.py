import json

def extract_actions(input_filename, output_filename):
    actions = []
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除行首尾空白
                clean_line = line.strip()
                
                # 檢查是否以 "Action:" 開頭
                if clean_line.startswith("Action:"):
                    # 擷取 "Action:" 之後的部分並再次去除空白
                    action_value = clean_line[len("Action:"):].strip()
                    actions.append(action_value)
        
        # 將 list 轉換成題目要求的 ["action1", "action2", ...] 格式字串
        # ensure_ascii=False 是為了確保如果有中文能正常顯示
        output_content = json.dumps(actions, ensure_ascii=False)
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(output_content)
            
        print(f"成功！已從 {input_filename} 提取 {len(actions)} 個動作。")
        print(f"結果已儲存至: {output_filename}")

    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {input_filename}")
    except Exception as e:
        print(f"發生錯誤：{e}")

# 使用範例
if __name__ == "__main__":
    # 請確保你的原始檔案名稱正確（例如叫 input.txt）
    extract_actions('log/game1-5_MC-DML_revised_2025-12-22_18:48:49', 'actions_only.txt')