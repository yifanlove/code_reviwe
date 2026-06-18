#!/usr/bin/env python3
"""
客服 FAQ 自动分类脚本
用途：对用户发来的问题进行自动分类，分配到对应的客服组
"""

import json
import os
import yaml
from openai import OpenAI

# ============================================================
# 配置加载
# ============================================================

# API 配置
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("未设置环境变量 DEEPSEEK_API_KEY，请先执行: export DEEPSEEK_API_KEY=your_key")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 提示词配置文件路径（与此脚本同目录）
PROMPTS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task1_prompts.yaml")


def _load_prompts_config() -> dict:
    """从 YAML 文件加载提示词配置，带缓存"""
    with open(PROMPTS_YAML, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 模块级加载一次
_config = _load_prompts_config()

# 从配置中提取运行时参数
VALID_CATEGORIES = _config["categories"]["valid"]
MAX_RETRIES = _config["retry"]["max_attempts"]
FALLBACK_CATEGORY = _config["retry"]["fallback_category"]


def _build_system_prompt(config: dict) -> str:
    """根据 YAML 配置组装完整的系统提示词"""
    sp = config["system_prompt"]

    # -- 角色 --
    prompt = f"你是一个{sp['role']}。你的任务是{sp['task']}。\n\n"

    # -- 分类定义 --
    prompt += sp["category_definitions"].strip() + "\n\n"

    # -- 分类规则 --
    prompt += sp["classification_rules"].strip() + "\n\n"

    # -- Few-Shot 示例 --
    prompt += sp["few_shot_template"].strip() + "\n\n"
    for i, example in enumerate(sp["few_shot_examples"], start=1):
        prompt += f"示例{i}:\n"
        prompt += f"用户问题：{example['user']}\n"
        prompt += f"分类结果：{example['result']}\n\n"

    # -- 输出要求 --
    prompt += sp["output_requirements"].strip()

    return prompt


def _get_user_message_template(config: dict) -> str:
    """获取用户消息模板"""
    return config["system_prompt"]["user_message_template"]


def _get_retry_correction_template(config: dict) -> str:
    """获取重试纠正消息模板"""
    return config["system_prompt"]["retry_correction_template"]


# 构建系统提示词
SYSTEM_PROMPT = _build_system_prompt(_config)
USER_MESSAGE_TEMPLATE = _get_user_message_template(_config)
RETRY_CORRECTION_TEMPLATE = _get_retry_correction_template(_config)


def classify_question(question: str) -> str:
    """对单条用户问题进行分类，带验证重试机制"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE_TEMPLATE.format(question=question)}
    ]

    for attempt in range(MAX_RETRIES):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # 验证结果是否在有效分类中
        if result in VALID_CATEGORIES:
            return result

        # 结果无效，将错误结果和纠正提示加入对话历史，继续重试
        messages.append({"role": "assistant", "content": result})
        messages.append({
            "role": "user",
            "content": RETRY_CORRECTION_TEMPLATE.format(
                result=result,
                categories="、".join(VALID_CATEGORIES)
            )
        })

    # 超过最大重试次数仍未得到有效结果，兜底返回配置的兜底类别
    return FALLBACK_CATEGORY


def batch_classify(input_file: str, output_file: str):
    """批量分类"""
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    results = []
    for item in questions:
        question = item['question']
        category = classify_question(question)
        results.append({
            'id': item['id'],
            'question': question,
            'predicted_category': category
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"分类完成，共处理 {len(results)} 条问题")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python classifier.py <输入文件> <输出文件>")
        sys.exit(1)

    batch_classify(sys.argv[1], sys.argv[2])
