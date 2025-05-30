# Prompt优化Pipeline

这是一个用于医疗诊断prompt优化的pipeline项目。该项目通过不断调整prompt来提高大模型在医疗诊断任务上的准确性。

## 项目结构

```
.
├── main.py                 # 主程序入口
├── prompt_optimizer.py     # Prompt优化器实现
├── api_clients.py          # API客户端实现
├── data/                   # 数据目录
│   ├── example_data.json  # 示例医疗数据
│   └── ground_truth.json  # 真实疾病标签
└── results/               # 结果输出目录
```

## 环境要求

- Python 3.8+
- 相关Python包（见requirements.txt）
- DeepSeek API密钥
- Qwen API密钥

## 安装

1. 克隆项目并安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
创建.env文件并添加以下内容：
```
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_API_URL=https://api.deepseek.com/v1
QWEN_API_KEY=your_qwen_api_key
QWEN_API_URL=https://api.qwen.ai/v1
```

## 使用方法

1. 准备数据：
   - 在data/example_data.json中放入示例医疗数据
   - 在data/ground_truth.json中放入对应的真实疾病标签

2. 运行优化pipeline：
```bash
python main.py
```

3. 查看结果：
   - 优化后的最佳prompt将保存在results/best_prompt.txt中

## 工作流程

1. 使用DeepSeek API生成优化的prompt
2. 将优化后的prompt输入到Qwen中进行疾病预测
3. 计算预测结果与真实标签的重合度
4. 重复以上步骤直到达到完全重合或达到最大迭代次数
