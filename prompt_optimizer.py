from typing import Dict, List
import json
import re
import logging
from api_clients import DeepSeekAPI, QwenAPI

class PromptOptimizer:
    def __init__(self, deepseek_optimizer: DeepSeekAPI, deepseek_analyzer: DeepSeekAPI, qwen_client: QwenAPI):
        self.deepseek_optimizer = deepseek_optimizer  # Prompt优化器
        self.deepseek_analyzer = deepseek_analyzer    # 差异分析器
        self.qwen_client = qwen_client               # 疾病预测器
        self.current_iteration = 0
        self.optimization_history = []  # 记录每次迭代的结果
        self.logger = logging.getLogger()
        
        self.base_prompt_template = """你是一位拥有30年临床经验的资深医生，精通各种医学影像的解读和疾病诊断。请严格按照以下要求分析患者的影像报告：

【任务说明】
仔细分析输入的医学影像数据，识别所有异常描述，结合临床经验判断可能的疾病。你需要综合考虑各项指标之间的相互关系，而不是孤立地看待单个异常值。

【重要规则】
1. 只输出有明确异常支持的疾病诊断
2. 如果指标在正常范围内或异常程度轻微，不作为诊断依据
3. 优先考虑常见病、多发病
4. 使用标准ICD-10疾病命名，但一定不要输出icd-10编码，只许输出疾病名称
5. 必须保证输出格式不变

【输出要求】
1. 必须以 JSON 格式输出，包含 diseases 数组
2. 不要输出任何解释性文字

【输出格式】
{
    "diseases": ["疾病1", "疾病2", "疾病3"]
}"""

    def log_iteration_results(self, iteration_data: Dict):
        """记录每次迭代的详细信息"""
        self.logger.info("\n" + "="*50)
        self.logger.info(f"迭代 {self.current_iteration} 详细信息")
        self.logger.info("="*50)
        
        # 1. 记录优化前后的prompt差异
        self.logger.info("\n【Prompt优化差异】")
        self.logger.info("-"*30)
        self.logger.info("优化前：")
        old_rules = self.extract_rules(iteration_data['old_prompt'])
        self.logger.info(old_rules)
        self.logger.info("\n优化后：")
        new_rules = self.extract_rules(iteration_data['new_prompt'])
        self.logger.info(new_rules)
        
        # 2. 记录每个案例的预测结果和分析
        self.logger.info("\n【案例分析】")
        for result in iteration_data['analysis_results']:
            self.logger.info(f"\n案例 {result['case_id']}:")
            self.logger.info("-"*30)
            self.logger.info(f"预测疾病: {result['predictions']}")
            self.logger.info(f"实际疾病: {result['ground_truth']}")
            self.logger.info(f"准确率: {result['differences']['accuracy']:.2%}")
            self.logger.info(f"精确率: {result['differences']['precision']:.2%}")
            
            # 记录DS分析器的分析结果
            self.logger.info("\n问题分析：")
            self.logger.info(result['analysis_report'])
        
        # 3. 记录总体统计信息
        self.logger.info("\n【本轮总结】")
        self.logger.info("-"*30)
        summary = iteration_data['summary']
        self.logger.info(f"总案例数: {summary['total_cases']}")
        self.logger.info(f"常见漏诊: {summary['common_missed_diagnoses']}")
        self.logger.info(f"常见误诊: {summary['common_wrong_diagnoses']}")
        self.logger.info("="*50 + "\n")

    def extract_rules(self, prompt: str) -> str:
        """从prompt中提取重要规则部分"""
        rules_start = prompt.find("【重要规则】")
        rules_end = prompt.find("【输出要求】")
        if rules_start != -1 and rules_end != -1:
            return prompt[rules_start:rules_end].strip()
        return "未找到规则部分"

    def analyze_differences(self, predicted: List[str], ground_truth: List[str]) -> Dict:
        """分析预测结果和真实标签之间的差异"""
        # 标准化处理
        pred_set = {d.strip('.,，。 \t').lower() for d in predicted if d}
        truth_set = {d.strip('.,，。 \t').lower() for d in ground_truth if d}
        
        # 计算差异
        missed = truth_set - pred_set  # 漏诊
        extra = pred_set - truth_set   # 误诊
        correct = pred_set & truth_set # 正确诊断
        
        return {
            'missed_diagnoses': list(missed),
            'wrong_diagnoses': list(extra),
            'correct_diagnoses': list(correct),
            'accuracy': len(correct) / max(len(truth_set), 1) if truth_set else 0,
            'precision': len(correct) / max(len(pred_set), 1) if pred_set else 0
        }

    def extract_diseases_from_response(self, response: str) -> List[str]:
        """从模型响应中提取疾病名称列表"""
        try:
            # 移除可能的Markdown代码块标记
            cleaned_response = response.strip()
            if cleaned_response.startswith("```"):
                # 找到第一个和最后一个```的位置
                first_marker = cleaned_response.find("```")
                if first_marker != -1:
                    # 跳过第一行，找到内容的开始
                    content_start = cleaned_response.find("\n", first_marker) + 1
                    # 从内容开始处往后找结束标记
                    end_marker = cleaned_response.find("```", content_start)
                    if end_marker != -1:
                        # 提取内容部分（不包含Markdown标记）
                        cleaned_response = cleaned_response[content_start:end_marker].strip()
            
            # 尝试解析处理后的响应
            result = json.loads(cleaned_response)
            if isinstance(result, dict):
                # 检查是否有错误信息
                if 'error' in result:
                    self.logger.warning(f"API返回错误: {result['error']}")
                    if 'raw_response' in result:
                        self.logger.debug(f"原始响应: {result['raw_response']}")
                    return []
                
                # 提取疾病列表
                if 'diseases' in result:
                    diseases = result['diseases']
                    if isinstance(diseases, list):
                        # 过滤并清理疾病名称
                        return [str(disease).strip() for disease in diseases if disease and str(disease).strip()]
                    elif diseases:  # 如果是单个字符串
                        return [str(diseases).strip()]
            
            self.logger.warning(f"无效的响应格式: {response}")
            return []
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON解析错误: {str(e)}\n原始响应: {response}")
            return []
        except Exception as e:
            self.logger.error(f"处理响应时出错: {str(e)}")
            return []

    def analyze_results(self, optimized_prompt: str, case_data: Dict) -> Dict:
        """使用第二个DeepSeek分析优化结果"""
        # 让 Qwen 预测疾病
        predicted = self.predict_diseases(optimized_prompt, case_data['input_data'])
        gt_diseases = case_data['gt_data'].get('diseases', [])
        differences = self.analyze_differences(predicted, gt_diseases)
        
        # 构造分析上下文
        analysis_context = {
            'base_prompt': optimized_prompt,
            'input_data': case_data['input_data'],
            'predictions': predicted,
            'ground_truth': gt_diseases,
            'differences': differences
        }
        
        # 让DS分析器分析差异原因
        analysis_report = self.deepseek_analyzer.analyze_results(analysis_context)
        
        return {
            'predictions': predicted,
            'ground_truth': gt_diseases,
            'differences': differences,
            'analysis_report': analysis_report  # 统一使用 analysis_report
        }

    def generate_prompt(self, context: Dict) -> str:
        """完整的prompt优化流程
        
        工作流程：
        1. DS优化器：优化prompt的【重要规则】部分
        2. Qwen：使用优化后的prompt预测疾病
        3. DS分析器：分析预测结果与GT的差异，生成分析报告
        4. DS优化器：根据分析报告进行下一轮优化
        """
        try:
            # 验证必要参数
            if not context:
                self.logger.error("上下文参数为空")
                return self.base_prompt_template
                
            self.current_iteration = context.get('iteration', self.current_iteration + 1)
            base_prompt = context.get('base_prompt', self.base_prompt_template)
            all_cases = context.get('all_cases', [])
            
            if not all_cases:
                self.logger.warning("没有提供测试案例，无法进行优化")
                return base_prompt
            
            # 如果是第一轮迭代，直接优化prompt
            if self.current_iteration == 1:
                optimized_prompt = self.deepseek_optimizer.generate_prompt({
                    'base_prompt': base_prompt,
                    'iteration': self.current_iteration,
                    'previous_analysis': None
                })
            else:
                # 获取上一轮的分析结果
                if not self.optimization_history:
                    self.logger.warning("找不到上一轮的优化历史，将使用基础prompt")
                    return base_prompt
                    
                last_iteration = self.optimization_history[-1]
                
                # 验证历史数据的完整性
                if not all(key in last_iteration for key in ['analysis_results', 'new_prompt', 'summary']):
                    self.logger.error("上一轮的优化数据不完整")
                    return base_prompt
                
                # 构造case_analyses列表
                case_analyses = []
                for result in last_iteration['analysis_results']:
                    try:
                        case_analyses.append({
                            'case_id': result['case_id'],
                            'predictions': result['predictions'],
                            'ground_truth': result['ground_truth'],
                            'missed_diagnoses': result['differences']['missed_diagnoses'],
                            'wrong_diagnoses': result['differences']['wrong_diagnoses'],
                            'analysis_report': result['analysis_report']
                        })
                    except KeyError as e:
                        self.logger.error(f"分析结果数据结构不完整: {str(e)}")
                        continue
                
                if not case_analyses:
                    self.logger.warning("没有有效的案例分析结果，将使用基础prompt")
                    return base_prompt
                
                # 构造符合DeepSeekAPI预期的previous_analysis结构
                previous_analysis = {
                    'prompt': last_iteration['new_prompt'],
                    'case_analyses': case_analyses,
                    'analysis_summary': last_iteration['summary']
                }
                
                # 将上一轮的分析结果传给优化器
                optimized_prompt = self.deepseek_optimizer.generate_prompt({
                    'base_prompt': base_prompt,
                    'iteration': self.current_iteration,
                    'previous_analysis': previous_analysis
                })
            
            # 验证优化后的prompt
            if not optimized_prompt or not isinstance(optimized_prompt, str):
                self.logger.error("生成的prompt无效")
                return base_prompt
                
            if "【重要规则】" not in optimized_prompt:
                self.logger.error("优化后的prompt缺少必要的规则部分")
                return base_prompt
            
            # 2. 对每个案例进行预测和分析
            analysis_results = []
            total_missed = set()
            total_wrong = set()
            
            for case in all_cases:
                try:
                    # 验证案例数据的完整性
                    if not all(key in case for key in ['case_id', 'input_data', 'gt_data']):
                        self.logger.error(f"案例数据不完整: {case.get('case_id', '未知ID')}")
                        continue
                        
                    # 让Qwen预测疾病
                    case_analysis = self.analyze_results(optimized_prompt, case)
                    
                    # 验证分析结果
                    if not case_analysis or 'differences' not in case_analysis:
                        self.logger.error(f"案例分析结果无效: {case['case_id']}")
                        continue
                        
                    # DS分析器分析结果
                    analysis_results.append({
                        'case_id': case['case_id'],
                        **case_analysis
                    })
                    
                    total_missed.update(case_analysis['differences']['missed_diagnoses'])
                    total_wrong.update(case_analysis['differences']['wrong_diagnoses'])
                    
                except Exception as e:
                    self.logger.error(f"处理案例时出错 {case.get('case_id', '未知ID')}: {str(e)}")
                    continue
            
            if not analysis_results:
                self.logger.error("没有成功分析任何案例")
                return base_prompt
            
            # 3. 记录本次迭代结果
            iteration_data = {
                'iteration': self.current_iteration,
                'old_prompt': base_prompt,
                'new_prompt': optimized_prompt,
                'analysis_results': analysis_results,
                'summary': {
                    'total_cases': len(all_cases),
                    'common_missed_diagnoses': list(total_missed),
                    'common_wrong_diagnoses': list(total_wrong)
                }
            }
            
            # 记录详细日志
            self.log_iteration_results(iteration_data)
            
            # 保存到历史记录，供下一轮使用
            self.optimization_history.append(iteration_data)
            
            return optimized_prompt
            
        except Exception as e:
            self.logger.error(f"生成prompt时出错: {str(e)}")
            self.logger.error("详细错误信息:", exc_info=True)
            return base_prompt

    def predict_diseases(self, prompt: str, data: Dict, max_retries: int = 3) -> List[str]:
        """使用Qwen预测疾病"""
        for attempt in range(max_retries):
            try:
                response = self.qwen_client.predict(prompt, data)
                diseases = self.extract_diseases_from_response(response)
                if diseases:  # 如果成功获取到疾病列表
                    return diseases
                self.logger.warning(f"尝试 {attempt + 1}/{max_retries}: 模型未返回任何疾病")
            except Exception as e:
                self.logger.error(f"尝试 {attempt + 1}/{max_retries}: 预测疾病时出错: {str(e)}")
                if attempt == max_retries - 1:  # 最后一次尝试
                    return []
        return []

    def evaluate_optimization_progress(self) -> Dict:
        """评估优化进展情况"""
        if not self.optimization_history:
            return {
                'status': 'no_progress',
                'message': '还未开始优化',
                'best_iteration': 0,
                'best_accuracy': 0.0
            }
        
        best_accuracy = 0.0
        best_iteration = 0
        accuracy_trend = []
        
        for i, iteration in enumerate(self.optimization_history, 1):
            iteration_accuracy = 0.0
            total_cases = 0
            
            for result in iteration['analysis_results']:
                if 'differences' in result and 'accuracy' in result['differences']:
                    iteration_accuracy += result['differences']['accuracy']
                    total_cases += 1
            
            avg_accuracy = iteration_accuracy / total_cases if total_cases > 0 else 0.0
            accuracy_trend.append(avg_accuracy)
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_iteration = i
        
        # 分析优化趋势
        is_improving = len(accuracy_trend) >= 2 and accuracy_trend[-1] > accuracy_trend[-2]
        is_stagnating = len(accuracy_trend) >= 3 and all(
            abs(accuracy_trend[i] - accuracy_trend[i-1]) < 0.01  # 1%的改进阈值
            for i in range(len(accuracy_trend)-1, len(accuracy_trend)-3, -1)
        )
        
        status = 'improving' if is_improving else 'stagnating' if is_stagnating else 'unstable'
        message = {
            'improving': '优化效果持续提升中',
            'stagnating': '优化效果已趋于稳定',
            'unstable': '优化效果不稳定，需要调整策略'
        }[status]
        
        return {
            'status': status,
            'message': message,
            'best_iteration': best_iteration,
            'best_accuracy': best_accuracy,
            'accuracy_trend': accuracy_trend
        }
