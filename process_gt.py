import os
import json
import pandas as pd
from typing import Dict, Any

def process_excel_data(excel_path: str) -> None:
    """
    处理Excel文件，提取数据并保存为JSON文件
    
    Args:
        excel_path: Excel文件的路径
    """
    try:
        # 读取Excel文件
        print(f"正在读取Excel文件: {excel_path}")
        df = pd.read_excel(excel_path)
        
        # 确保必要的列存在
        required_columns = ['ID号', 'content', 'gt', 'pred']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Excel文件中缺少以下列: {', '.join(missing_columns)}")
        
        # 创建输出目录
        os.makedirs(os.path.join('data', 'inputs'), exist_ok=True)
        os.makedirs(os.path.join('data', 'gts'), exist_ok=True)
        
        # 处理每一行数据
        processed_count = 0
        for _, row in df.iterrows():
            case_id = str(row['ID号']).strip()
            if not case_id:  # 跳过空ID
                continue
                
            # 准备gt数据
            gt_data = {
                'diseases': [d.strip() for d in str(row['gt']).split(',')] if pd.notna(row['gt']) else []
            }
            
            # 准备输入数据
            input_data = {
                'content': str(row['content']) if pd.notna(row['content']) else ""
            }
            
            # 保存gt文件
            gt_path = os.path.join('data', 'gts', f'{case_id}_gt.json')
            with open(gt_path, 'w', encoding='utf-8') as f:
                json.dump(gt_data, f, ensure_ascii=False, indent=2)
            
            # 保存输入文件
            input_path = os.path.join('data', 'inputs', f'{case_id}_exam_input.json')
            with open(input_path, 'w', encoding='utf-8') as f:
                json.dump(input_data, f, ensure_ascii=False, indent=2)
            
            processed_count += 1
            
        print(f"\n数据处理完成！")
        print(f"共处理 {processed_count} 条记录")
        print(f"输出文件保存在：")
        print(f"- data/gts/")
        print(f"- data/inputs/")
        
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        raise

if __name__ == "__main__":
    excel_path = "影像数据集.xlsx"
    process_excel_data(excel_path)
