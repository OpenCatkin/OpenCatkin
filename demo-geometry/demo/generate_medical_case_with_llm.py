import ollama
import json
import random

def generate_medical_case_with_llm(n_cases: int = 500) -> List[Dict]:
    """
    使用 Qwen3:8B 生成高质量医疗病例
    """
    system_prompt = """你是一个专业的医疗数据生成助手。请生成真实的门诊病例。

疾病列表（只能从这里选）：
- common_cold (普通感冒)
- influenza (流感)
- covid_19 (新冠)
- pneumonia (肺炎)
- bronchitis (支气管炎)
- stomach_flu (肠胃炎)
- allergy (过敏)
- migraine (偏头痛)

症状列表（只能从这里选）：
fever, cough, headache, fatigue, sore_throat, runny_nose, shortness_of_breath, chest_pain, nausea, vomiting, diarrhea, rash, muscle_pain

输出格式要求：JSON 列表，每个元素包含 {"symptoms": ["symptom1", "symptom2"], "disease": "disease_name"}

不要输出任何 Markdown 标记，只输出纯 JSON。"""

    all_cases = []
    
    print(f"🚀 开始使用 Qwen3:8B 生成 {n_cases} 个医疗病例...")
    
    for i in range(n_cases):
        try:
            response = ollama.generate(
                model="qwen2.5:7b", # 或者 qwen3:8b，根据你本地的模型名调整
                prompt=f"请生成 1 个真实的门诊病例。",
                system=system_prompt,
                format="json",
                options={"temperature": 0.8, "num_predict": 200}
            )
            
            # 解析结果
            result_text = response["response"].strip()
            # 有时候 LLM 会包裹在 ```json ... ``` 里，尝试清理
            if result_text.startswith("```"):
                result_text = result_text.split("\n", 1)[1].rsplit("\n", 1)[0]
            
            case = json.loads(result_text)
            
            # 如果返回的是列表，取第一个；如果是单个字典，直接用
            if isinstance(case, list) and len(case) > 0:
                case = case[0]
            
            # 验证数据合法性
            if "symptoms" in case and "disease" in case:
                all_cases.append(case)
            
            if (i + 1) % 50 == 0:
                print(f"   已生成 {i+1}/{n_cases} 个病例")
                
        except Exception as e:
            print(f"   生成第 {i+1} 个病例时出错: {e}，跳过...")
            continue
    
    print(f"✅ 成功生成 {len(all_cases)} 个有效病例！")
    return all_cases

# 使用示例（运行一次即可，生成后保存到文件）
if __name__ == "__main__":
    # llm_cases = generate_medical_case_with_llm(300)
    # with open("llm_medical_cases.json", "w", encoding="utf-8") as f:
    #     json.dump(llm_cases, f, ensure_ascii=False, indent=2)
    pass