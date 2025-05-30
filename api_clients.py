import time
from typing import Dict
import json
import logging
from openai import OpenAI

def center_log(logger, text: str, width: int = 80, char: str = "="):
    """创建居中的日志标题"""
    text = f" {text} "
    padding = (width - len(text)) // 2
    line = char * padding + text + char * padding
    if len(line) < width:  # 如果总长度为奇数，补充一个字符
        line += char
    logger.info("\n" + line + "\n")

# API配置
DEEPSEEK_API_KEY = "sk-izL-0DIrAPzFt4KIZFJ5xg"
DEEPSEEK_API_URL = "https://litellm.shukun.net/v1"

# TODO: 获取到Qwen API后替换这里的值
# Qwen 2.5
QWEN_API_KEY = "123"
QWEN_API_URL = "http://10.20.4.86:30001/v1"
# Qwen 3
# QWEN_API_KEY = "sk-FBx4oHDxpl6b9DjYdVsLog"
# QWEN_API_URL = "http://10.20.0.201:30509/v1"

class DeepSeekAPI:
    def __init__(self, role="optimizer"):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_URL
        )
        self.role = role  # 可以是 "optimizer" 或 "analyzer"
        self.logger = logging.getLogger()

    def _make_api_call(self, messages, retries=3):
        """处理API调用，包括重试逻辑"""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-v3-aliyun",
                    messages=messages,
                    temperature=0.7,
                    stream=False
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == retries - 1:
                    raise
                self.logger.warning(f"API调用失败 (尝试 {attempt + 1}/{retries}): {str(e)}")
        return None

    def generate_prompt(self, context: Dict) -> str:
        """优化prompt的DeepSeek实例使用此方法"""
        try:
            system_message = """你是一个专业的医疗prompt优化专家。你必须严格按照以下规则工作：

1. 格式控制（最重要）:
- 你只能修改【重要规则】部分的内容
- 其他部分（【任务说明】、【输出要求】、【输出格式】）必须保持完全一致
- 禁止添加、删除或修改任何分隔符【】
- 所有格式标记必须使用中文方括号【】

2. 规则优化要求:
- 每条规则必须明确、具体、可执行
- 规则必须包含明确的判断标准和阈值
- 禁止在规则中包含任何ICD编码
- 避免使用模糊的描述词

3. 输出规范：
- 必须确保优化后的规则符合JSON输出格式要求
- 禁止添加任何会破坏JSON结构的内容
- 疾病名称必须是标准化的中文名称

你的输出必须严格保持以下格式：

【任务说明】
(保持原文不变)

【重要规则】
(这里是你唯一可以修改的部分)

【输出要求】
(保持原文不变)

【输出格式】
(保持原文不变)"""

            base_prompt = context.get('base_prompt', '')
            iteration = context.get('iteration', 1)
            previous_analysis = context.get('previous_analysis', None)

            # 构建用户消息
            if previous_analysis is None:
                user_message = f"""这是第 {iteration} 轮优化。

当前prompt:
{base_prompt}

请严格按照当前prompt的格式优化【重要规则】部分，确保：
1. 只修改【重要规则】部分
2. 其他部分保持完全一致
3. 优化后的规则更加明确和可执行"""
            else:
                user_message = f"""这是第 {iteration} 轮优化。

分析结果：
{json.dumps(previous_analysis, ensure_ascii=False, indent=2)}

当前prompt：
{base_prompt}

请基于分析结果优化【重要规则】部分，确保：
1. 只修改【重要规则】部分
2. 其他部分保持完全一致
3. 新规则必须解决分析中发现的问题"""

            content = self._make_api_call([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ])

            # 验证返回的格式
            if not all(section in content for section in ["【任务说明】", "【重要规则】", "【输出要求】", "【输出格式】"]):
                self.logger.warning("API返回的prompt格式不完整，使用原prompt")
                return base_prompt

            # 验证格式是否正确
            sections = content.split("【")
            if len(sections) != 5:  # 开头部分 + 4个标题部分
                self.logger.warning("API返回的prompt段落数量不正确，使用原prompt")
                return base_prompt

            return content

        except Exception as e:
            self.logger.error(f"生成prompt失败: {str(e)}")
            return context.get('base_prompt', '')

    def analyze_results(self, context: Dict) -> str:
        """分析器DeepSeek实例使用此方法分析预测结果"""
        try:
            system_message = """你是一个专业的医疗诊断分析专家。
你的任务是分析模型预测结果与真实标签之间的差异，为优化器提供具体的改进建议。

分析重点：
1. 对比预测结果和真实结果的差异
2. 找出漏诊和误诊的具体原因
3. 提供明确的改进建议

输出格式要求：
1. 分析报告必须包含具体的数字和案例
2. 建议必须可执行，有明确的判断标准
3. 重点关注影响准确率的主要问题"""

            differences = context.get('differences', {})
            predictions = context.get('predictions', [])
            ground_truth = context.get('ground_truth', [])
            input_data = context.get('input_data', {})

            user_message = f"""请分析以下诊断结果：

输入数据: {json.dumps(input_data, ensure_ascii=False, indent=2)}
预测疾病: {predictions}
实际疾病: {ground_truth}

差异分析:
- 漏诊: {differences.get('missed_diagnoses', [])}
- 误诊: {differences.get('wrong_diagnoses', [])}
- 准确率: {differences.get('accuracy', 0):.2%}"""

            return self._make_api_call([
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ])

        except Exception as e:
            self.logger.error(f"分析结果失败: {str(e)}")
            return "分析失败"

class QwenAPI:
    def __init__(self):
        self.client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_API_URL
        )
        self.logger = logging.getLogger("Qwen")
        self.file_logger = logging.getLogger('file_logger')  # 用于详细日志

    def predict(self, prompt: str, data: Dict) -> str:
        """调用API进行疾病预测"""
        try:
            # 记录接收到的原始prompt（仅在日志文件中）
            center_log(self.file_logger, "收到的原始Prompt")
            self.file_logger.info(prompt)
            
            # 检查原始prompt是否已经包含JSON格式说明
            if "diseases" not in prompt:
                format_instruction = """
请严格按照以下JSON格式返回结果：
{
    "diseases": [
        "疾病1",
        "疾病2"
    ]
}

格式要求：
1. 必须是合法的JSON格式
2. diseases数组必须存在，即使为空也要保留
3. 每个疾病名称必须是非空字符串
4. 不要在JSON外添加任何注释或说明
5. 不要使用Markdown代码块

示例输出：
{"diseases": ["支气管炎", "肺气肿"]}"""
                enhanced_prompt = f"{prompt}\n\n{format_instruction}"
            else:
                enhanced_prompt = prompt
            
            # 记录发送给API的完整prompt（仅在日志文件中）
            center_log(self.file_logger, "发送给API的完整Prompt")
            self.file_logger.info(enhanced_prompt)

            try:
                response = self.client.chat.completions.create(
                    model="Qwen2.5-32B-Instruct-GPTQ-Int4",
                    messages=[
                        {
                            "role": "system",
                            "content": enhanced_prompt
                        },
                        {
                            "role": "user",
                            "content": json.dumps(data, ensure_ascii=False, indent=2)
                        }
                    ],
                    temperature=0.3,
                    stream=False
                )
            except Exception as e:
                self.logger.error(f"API调用失败: {str(e)}")
                return json.dumps({"diseases": []}, ensure_ascii=False, indent=2)
            
            content = response.choices[0].message.content.strip()
            
            # 记录API返回的原始内容（仅在日志文件中）
            center_log(self.file_logger, "API返回的原始内容")
            self.file_logger.info(content)
            
            # 清理响应内容：去除所有非JSON内容
            try:
                # 尝试直接解析
                result = json.loads(content)
            except json.JSONDecodeError:
                # 如果失败，尝试提取JSON部分
                import re
                json_pattern = r'\{[^{}]*"diseases"\s*:\s*\[[^\]]*\][^{}]*\}'
                matches = re.findall(json_pattern, content)
                if matches:
                    try:
                        result = json.loads(matches[0])
                    except:
                        self.logger.error("JSON提取失败")
                        return json.dumps({'diseases': []}, ensure_ascii=False, indent=2)
                else:
                    self.logger.error("未找到符合格式的JSON")
                    return json.dumps({'diseases': []}, ensure_ascii=False, indent=2)
            
            # 标准化输出
            if isinstance(result, dict) and 'diseases' in result:
                diseases = result['diseases']
                if isinstance(diseases, list):
                    diseases = [str(d).strip() for d in diseases if d and str(d).strip()]
                    return json.dumps({'diseases': diseases}, ensure_ascii=False, indent=2)
                elif diseases:
                    return json.dumps({'diseases': [str(diseases).strip()]}, ensure_ascii=False, indent=2)
            
            self.logger.error("返回数据格式不符合要求")
            return json.dumps({'diseases': []}, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            return json.dumps({'diseases': []}, ensure_ascii=False, indent=2)

class APIClient:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # 配置日志
        self.logger = logging.getLogger("APIClient")
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器 - 记录详细信息
        fh = logging.FileHandler("api_calls.log", encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        
        # 控制台处理器 - 只显示关键信息
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def make_request(self, messages: list, max_retries: int = 3) -> Dict:
        """发送API请求并处理响应"""
        center_log(self.logger, "开始新的API请求")
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"发送请求 (尝试 {attempt + 1}/{max_retries})")
                self.logger.debug(f"请求内容: {json.dumps(messages, ensure_ascii=False, indent=2)}")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                
                result = response.choices[0].message.content
                self.logger.debug(f"API响应: {result}")
                center_log(self.logger, "请求成功完成")
                
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    self.logger.error("响应不是有效的JSON格式")
                    return {"error": "Invalid JSON response", "raw_response": result}
                    
            except Exception as e:
                self.logger.error(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    center_log(self.logger, "所有重试均失败", char="!")
                    return {"error": str(e)}

    def clean_response(self, response: Dict) -> Dict:
        """清理和验证API响应"""
        if "error" in response:
            self.logger.error(f"处理错误响应: {response['error']}")
            return response
            
        self.logger.debug("开始清理响应...")
        # ...existing cleanup code...
        return response
