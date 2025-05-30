import json
import os
import logging
from datetime import datetime
from typing import List, Dict
from prompt_optimizer import PromptOptimizer
from api_clients import DeepSeekAPI, QwenAPI

def setup_logging():
    """配置日志系统"""
    logger = logging.getLogger("Main")
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler("process.log", encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def load_example_data(file_path: str) -> Dict:
    """加载JSON数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
            try:
                return json.loads(data)
            except json.JSONDecodeError as je:
                logger = logging.getLogger()
                logger.error(f"JSON解析错误 {file_path}: {str(je)}")
                logger.error(f"原始数据内容: {data}")
                raise
    except FileNotFoundError as fe:
        logger = logging.getLogger()
        logger.error(f"找不到文件 {file_path}")
        raise
    except Exception as e:
        logger = logging.getLogger()
        logger.error(f"加载文件失败 {file_path}: {str(e)}")
        logger.error("详细错误信息:", exc_info=True)
        raise

def process_input_data(input_data):
    """处理输入数据，确保返回疾病列表"""
    logger = logging.getLogger()
    try:
        if input_data is None:
            logger.error("输入数据为None")
            return []
            
        if isinstance(input_data, list):
            result = [str(d).strip() for d in input_data if d is not None]
            logger.debug(f"处理列表数据: {input_data} -> {result}")
            return result
            
        elif isinstance(input_data, dict):
            if 'diseases' in input_data:
                diseases = input_data['diseases']
                if isinstance(diseases, list):
                    result = [str(d).strip() for d in diseases if d is not None]
                    logger.debug(f"处理字典中的diseases列表: {diseases} -> {result}")
                    return result
                elif diseases is not None:
                    result = [str(diseases).strip()]
                    logger.debug(f"处理字典中的单个disease: {diseases} -> {result}")
                    return result
            
            logger.warning(f"字典中没有找到diseases键: {input_data}")
            return []
            
        else:
            logger.warning(f"未知的输入数据类型: {type(input_data)}")
            return []
            
    except Exception as e:
        logger.error(f"处理输入数据时出错: {str(e)}", exc_info=True)
        logger.error(f"问题数据: {input_data}")
        return []

def calculate_overlap(predicted_diseases: List[str], ground_truth: List[str]) -> float:
    """计算预测疾病列表和真实疾病列表的重合度"""
    if not predicted_diseases or not ground_truth:
        return 0.0
    
    # 将疾病名称标准化
    pred_set = {d.strip('.,，。 \t').lower() for d in predicted_diseases if d}
    truth_set = {d.strip('.,，。 \t').lower() for d in ground_truth if d}
    
    if not pred_set or not truth_set:
        return 0.0
        
    overlap = pred_set & truth_set
    union = pred_set | truth_set
    
    # 记录重合和差异
    overlap_diseases = list(overlap)
    missed_diseases = list(truth_set - pred_set)
    extra_diseases = list(pred_set - truth_set)
    
    return {
        'score': len(overlap) / len(union),
        'overlap_diseases': overlap_diseases,
        'missed_diseases': missed_diseases,
        'extra_diseases': extra_diseases
    }

def main():
    logger = setup_logging()
    logger.info("\n" + "="*60)
    logger.info("医疗影像诊断prompt优化系统")
    logger.info("="*60 + "\n")
    
    logger.info("【初始化阶段】")
    logger.info("-"*40)
    
    # 初始化API客户端和优化器
    logger.info("⚡ 正在初始化API客户端...")
    try:
        deepseek_optimizer = DeepSeekAPI(role="optimizer")
        logger.info("✓ DeepSeek优化器API就绪 (用途：生成优化后的prompt)")
        
        deepseek_analyzer = DeepSeekAPI(role="analyzer")
        logger.info("✓ DeepSeek分析器API就绪 (用途：分析优化结果)")
        
        qwen_client = QwenAPI()
        logger.info("✓ 通义千问API就绪 (用途：执行疾病预测)")
        
        optimizer = PromptOptimizer(deepseek_optimizer, deepseek_analyzer, qwen_client)
        logger.info("✓ Prompt优化器初始化完成")
    except Exception as e:
        logger.error(f"✗ API初始化失败: {str(e)}")
        raise
    
    logger.info("\n【数据加载阶段】")
    logger.info("-"*40)
    
    # 获取所有输入文件
    input_dir = os.path.join('data', 'inputs')
    input_files = [f for f in os.listdir(input_dir) if f.endswith('_exam_input.json')]
    
    # 创建results目录
    os.makedirs('results', exist_ok=True)
    
    # 确保logs目录存在
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # 创建本次运行的日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'optimization_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    max_iterations = 10
    best_prompt = optimizer.base_prompt_template
    best_avg_overlap = 0.0
    stagnation_count = 0  # 连续停滞次数
    last_best_accuracy = 0.0  # 上一次最佳准确率
    
    # 预先加载所有数据
    logger.info(f"正在从 {input_dir} 加载测试案例...")
    all_cases = []
    for input_file in input_files:
        case_id = input_file.replace('_exam_input.json', '')
        try:
            input_path = os.path.join(input_dir, input_file)
            gt_path = os.path.join('data', 'gts', f'{case_id}_gt.json')
            
            case_data = {
                'case_id': case_id,
                'input_data': load_example_data(input_path),
                'gt_data': load_example_data(gt_path)
            }
            all_cases.append(case_data)
            logger.info(f"✓ 案例 {case_id:<15} 数据完整性检查通过")
        except Exception as e:
            logger.error(f"✗ 案例 {case_id:<15} 数据加载失败: {str(e)}")
            continue
    
    logger.info(f"\n>>> 数据加载完成，共 {len(all_cases)} 个有效案例")
    
    logger.info("\n【优化迭代阶段】")
    logger.info("-"*40)
    
    for i in range(max_iterations):
        logger.info("\n" + "="*60)
        logger.info(f"迭代轮次: {i+1}/{max_iterations}")
        logger.info(f"当前最佳重合度: {best_avg_overlap:.2%}")
        logger.info("="*60 + "\n")
        
        logger.info("1️⃣ 正在生成新的prompt...")
        try:
            optimization_context = {
                'base_prompt': best_prompt,
                'all_cases': all_cases[:3],  # 先用前三个案例优化
                'iteration': i + 1,
                'previous_results': {
                    'best_overlap': best_avg_overlap,
                    'iteration': i
                }
            }
            
            new_prompt = optimizer.generate_prompt(optimization_context)
            logger.info("✓ 新prompt生成成功")
            
            # 检查优化进度
            progress = optimizer.evaluate_optimization_progress()
            logger.info(f"\n优化状态: {progress['message']}")
            logger.info(f"最佳准确率: {progress['best_accuracy']:.2%}")
            
            # 根据优化进度调整策略
            if progress['status'] == 'stagnating':
                stagnation_count += 1
                if stagnation_count >= 3:  # 连续3轮无显著改进
                    logger.info("\n🔄 检测到优化停滞，尝试调整策略...")
                    # 增加测试案例数量
                    all_cases_used = min(len(all_cases), 3 + stagnation_count)
                    optimization_context['all_cases'] = all_cases[:all_cases_used]
                    logger.info(f"增加测试案例数量至: {all_cases_used}")
            else:
                stagnation_count = 0
            
            if progress['status'] == 'improving':
                logger.info("📈 优化效果良好，继续当前策略")
            
            logger.info("\n【本轮prompt】")
            logger.info("-" * 40)
            logger.info(new_prompt)
            logger.info("-" * 40 + "\n")
            
        except Exception as e:
            logger.error(f"✗ 生成prompt失败: {str(e)}")
            new_prompt = best_prompt
        
        logger.info("2️⃣ 开始测试新prompt...")
        total_overlap = 0.0
        total_files = 0
        all_results = []
        
        for case in all_cases:
            case_id = case['case_id']
            input_data = case['input_data']
            gt_data = case['gt_data']
            
            try:
                predicted_diseases = optimizer.predict_diseases(new_prompt, input_data)
                processed_predictions = process_input_data(predicted_diseases)
                processed_gt = process_input_data(gt_data)
                
                overlap_result = calculate_overlap(processed_predictions, processed_gt)
                current_overlap = overlap_result['score']
                
                logger.info(f"\n案例 {case_id}:")
                logger.info(f"预测疾病：{', '.join(processed_predictions)}")
                logger.info(f"实际疾病：{', '.join(processed_gt)}")
                logger.info(f"重合程度：{current_overlap:.2%}")
                if overlap_result['missed_diseases']:
                    logger.info(f"漏诊疾病：{', '.join(overlap_result['missed_diseases'])}")
                if overlap_result['extra_diseases']:
                    logger.info(f"误诊疾病：{', '.join(overlap_result['extra_diseases'])}")
                
                total_overlap += current_overlap
                total_files += 1
                all_results.append({
                    'case_id': case_id,
                    'iteration': i + 1,
                    'predicted': processed_predictions,
                    'ground_truth': processed_gt,
                    'overlap': current_overlap
                })
                
            except Exception as e:
                logger.error(f"案例 {case_id} 处理失败: {str(e)}")
                continue
        
        avg_overlap = total_overlap / total_files if total_files > 0 else 0
        logger.info("\n" + "-" * 40)
        logger.info(f"本轮平均重合度：{avg_overlap:.2%}")
        
        if avg_overlap > best_avg_overlap:
            improvement = avg_overlap - best_avg_overlap
            best_avg_overlap = avg_overlap
            best_prompt = new_prompt
            logger.info(f"\n✨ 发现更优prompt! 提升了 {improvement:.2%}")
            logger.info(f"最佳重合度：{best_avg_overlap:.2%}")
            
            # 保存每个阶段的最佳prompt
            with open(f'results/best_prompt_{i+1}.txt', 'w', encoding='utf-8') as f:
                f.write(best_prompt)
            
            if avg_overlap == 1.0:
                logger.info("\n🎉 达到完全重合，提前结束优化!")
                break
        
        # 保存本轮结果
        results_file = os.path.join('logs', f'iteration_{i+1}_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'iteration': i + 1,
                'avg_overlap': avg_overlap,
                'best_avg_overlap': best_avg_overlap,
                'results': all_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info("\n3️⃣ DeepSeek分析器正在分析本轮结果...")
        # 这里可以添加分析器的具体实现
        logger.info("✓ 分析完成")
        
        # 检查是否需要提前结束
        if stagnation_count >= 5:  # 连续5轮无显著改进
            logger.info("\n🛑 检测到优化已经停滞，提前结束优化")
            break
    
    # 保存最终的最佳prompt
    with open('results/best_prompt.txt', 'w', encoding='utf-8') as f:
        f.write(best_prompt)
    
    # 生成优化报告
    optimization_report = {
        'total_iterations': i + 1,
        'best_accuracy': best_avg_overlap,
        'optimization_status': optimizer.evaluate_optimization_progress(),
        'timestamp': timestamp
    }
    
    with open(os.path.join('results', 'optimization_report.json'), 'w', encoding='utf-8') as f:
        json.dump(optimization_report, f, ensure_ascii=False, indent=2)
    
    logger.info("="*50)
    logger.info("优化完成!")
    logger.info(f"总迭代次数：{i + 1}")
    logger.info(f"最终最佳重合度：{best_avg_overlap:.2%}")
    logger.info(f"最佳prompt已保存至：results/best_prompt.txt")
    logger.info(f"完整优化日志已保存至：{log_file}")
    logger.info(f"优化报告已保存至：results/optimization_report.json")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger("Main")
        logger.error(f"程序运行时发生错误: {str(e)}", exc_info=True)
    finally:
        logger = logging.getLogger("Main")
        logger.info("\n程序运行结束")
