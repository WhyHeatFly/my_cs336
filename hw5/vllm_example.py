from vllm import LLM, SamplingParams

# 实例 Prompts
prompts = [
    "Hello, my name is",
    "The president of the united States is",
    "The capital of France is",
    "The future of AI is",
]

# 创建采样参数对象，在换行符时停止生成
sampling_params = SamplingParams(
    temperature = 1.0,
    top_p = 1.0,
    max_tokens = 1024,
    stop = ["\n"]
)

# 创建 LLM 实例
llm = LLM(model="./model/huggingface_cache")

# 生成文本 (输出是 RequestOutput 对象列表)
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    


