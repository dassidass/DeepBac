import json
import torch
import gc
import time
import pandas as pd
from unsloth import FastLanguageModel
import evaluate
from openai import OpenAI

# =====================================================================
# 1. Basic Evaluation Settings and Model Paths
# =====================================================================
# Place your DeepSeek API key here for LLM-as-a-Judge evaluation
DEEPSEEK_API_KEY = ""  # 👈 Put your key here

# List of fine-tuned model paths (ensure folder names are correct)
MODELS_TO_TEST = [
    "lora_deepbac_model_Qwen",
    "lora_deepbac_model_llama",
    "lora_deepbac_model_DeepSeek",
    "lora_deepbac_model_ALLaM"
]

TEST_FILE = "test_dataset.jsonl"
OUTPUT_FILE = "models_evaluation_results.csv"

# =====================================================================
# 2. Load Classical Evaluation Metrics (BLEU, ROUGE, BERTScore)
# =====================================================================
print("⏳ Loading evaluation metrics (BLEU, ROUGE, BERTScore)...")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

# Setup DeepSeek API (OpenAI compatible)
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def evaluate_with_llm_judge(question, reference, generated_answer):
    """
    Uses DeepSeek model as a 'Judge' to evaluate the answer semantically and pedagogically out of 10.
    """
    prompt = f"""
أنت أستاذ ومصحح جزائري صارم لاختبارات البكالوريا (مادة التاريخ والجغرافيا).
قم بتقييم إجابة الطالب مقارنة بـ "التصحيح النموذجي الوزاري".

سؤال الطالب: {question}
التصحيح النموذجي (المرجع): {reference}
إجابة الطالب (النموذج الذي يتم اختباره): {generated_answer}

قيّم الإجابة بناءً على معيارين فقط من 1 إلى 10:
1. الدقة المعرفية (Factual Accuracy): هل ذكر التواريخ، والأحداث، والأسماء بشكل صحيح ولم يخترع معلومات (هلوسة)؟ (علامة من 10).
2. التنسيق البيداغوجي (Format Compliance): هل الإجابة منظمة، واضحة، ومهيكلة بشكل يسهل قراءته؟ (علامة من 10).

أرجع النتيجة بتنسيق JSON فقط على الشكل التالي دون أي نص إضافي:
{{"Factual": 8.5, "Format": 9.0}}
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-v4-pro", # You can use deepseek-reasoner (R1) if available
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # Strict evaluation without creativity
            response_format={"type": "json_object"}
        )
        result_json = json.loads(response.choices[0].message.content)
        return float(result_json.get("Factual", 0)), float(result_json.get("Format", 0))
    except Exception as e:
        print(f"⚠️ Error in LLM Judge evaluation: {e}")
        return 0.0, 0.0

# =====================================================================
# 3. Load Test Questions
# =====================================================================
print(f"📂 Reading test questions from '{TEST_FILE}'...")
test_data = []
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        test_data.append(json.loads(line))
print(f"✅ Successfully loaded {len(test_data)} test questions.")

prompt_template = """أنت أستاذ جزائري خبير ومكلف بتصحيح امتحانات البكالوريا. مهمتك هي تقديم الإجابة النموذجية المباشرة والدقيقة لسؤال الطالب.

التعليمات الصارمة:
1. أجب باللغة العربية الفصحى فقط. يُمنع منعاً باتاً استخدام اللغة الصينية أو الإنجليزية أو أي لغة أخرى.
2. قدم الإجابة المطلوبة فقط وتوقف فوراً. لا تضف أي مقدمات أو خاتمات أو معلومات تاريخية خارج نطاق السؤال.

### السؤال:
{}

### الإجابة النموذجية:
"""

all_results = []

# =====================================================================
# 4. Start Model Evaluation Loop
# =====================================================================
print("\n" + "="*60)
print("🚀 Starting comprehensive model evaluation...")

for model_path in MODELS_TO_TEST:
    print(f"\n🔄 Evaluating model: {model_path}")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
        
        for q_idx, item in enumerate(test_data):
            question = item["instruction"]
            reference_answer = item["output"]
            
            print(f"   ❓ Question {q_idx + 1}/{len(test_data)}...")
            
            inputs = tokenizer([prompt_template.format(question)], return_tensors = "pt").to("cuda")

            start_time = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens = 512,
                use_cache = True,
                temperature = 0.3,
                repetition_penalty = 1.15,
                no_repeat_ngram_size = 3,
                pad_token_id = tokenizer.eos_token_id
            )
            generation_time = time.time() - start_time
            
            # Extract text
            response = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
            
            # ⬇️ Modification here: Use the exact phrase from the template to split ⬇️
            generated_answer = response.split("### الإجابة النموذجية:\n")[-1].strip()
            
            # --- Calculate Classical Metrics ---
            # 1. BLEU
            bleu_score = bleu_metric.compute(predictions=[generated_answer], references=[[reference_answer]])['bleu']
            
            # 2. ROUGE-L
            rouge_result = rouge_metric.compute(predictions=[generated_answer], references=[reference_answer])
            rouge_l = rouge_result['rougeL']
            
            # 3. BERTScore
            bert_result = bertscore_metric.compute(predictions=[generated_answer], references=[reference_answer], lang="ar")
            bert_score = sum(bert_result['f1']) / len(bert_result['f1']) # Average similarity
            
            # --- Modern Evaluation (LLM Judge) ---
            factual_score, format_score = evaluate_with_llm_judge(question, reference_answer, generated_answer)
            
            # Save results
            all_results.append({
                "Model": model_path,
                "Question": question,
                "Reference_Answer": reference_answer,
                "Generated_Answer": generated_answer,
                "BLEU": round(bleu_score, 4),
                "ROUGE-L": round(rouge_l, 4),
                "BERTScore_F1": round(bert_score, 4),
                "Judge_Factual (1-10)": factual_score,
                "Judge_Format (1-10)": format_score,
                "Generation_Time(s)": round(generation_time, 2)
            })
            
        # Free memory for the next model
        print(f"🧹 Freeing memory for model {model_path}...")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ An error occurred while evaluating model {model_path}: {e}")

# =====================================================================
# 5. Export and Save Final Results
# =====================================================================
print("\n" + "="*60)
print(f"💾 Saving comprehensive results to '{OUTPUT_FILE}'...")
df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

# Calculate and print averages for each model
print("\n📊 Averages Summary for Models:")
summary_df = df.groupby("Model").mean(numeric_only=True).round(3)
print(summary_df[["BLEU", "ROUGE-L", "BERTScore_F1", "Judge_Factual (1-10)", "Judge_Format (1-10)"]])

print("\n✅ Comprehensive evaluation process completed successfully!")