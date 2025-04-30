import os
import requests
from dotenv import load_dotenv
import gradio as gr
import time

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")

# 模型配置
MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "glm": "THUDM/GLM-Z1-9B-0414"
}


def chat_with_model(model_key, messages):
    """与指定模型进行交互"""
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODELS[model_key],
        "messages": messages,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API请求错误: {str(e)}")
        return "服务暂时不可用，请稍后再试。"


def gradio_chat(message, chat_history, model_key):
    """处理聊天交互（修复状态管理问题）"""
    # 构建对话历史（确保类型安全）
    formatted_history = []
    for entry in chat_history:
        if isinstance(entry, tuple) and len(entry) == 2:
            user, assistant = entry
            formatted_history.append((str(user), str(assistant)))

    # 准备API请求的消息
    messages = []
    for user_msg, assistant_msg in formatted_history:
        messages.extend([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ])
    messages.append({"role": "user", "content": str(message)})

    # 获取模型响应
    full_response = ""
    try:
        full_response = chat_with_model(model_key, messages)
    except Exception as e:
        full_response = f"请求出错：{str(e)}"

    # 流式输出处理
    chat_history.append((message, ""))  # 初始化空响应
    for char in full_response:
        last_query = chat_history[-1][0]
        chat_history[-1] = (last_query, chat_history[-1][1] + char)
        time.sleep(0.015)
        yield chat_history, ""


# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 多模型对话系统 (修复版)")

    # 使用State组件维护对话历史
    chat_state = gr.State([])

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=[
                ("DeepSeek-7B", "deepseek"),
                ("GLM-9B", "glm")
            ],
            value="deepseek",
            label="选择模型"
        )
        clear_btn = gr.Button("清空对话", variant="stop")

    chatbot = gr.Chatbot(
        label="对话历史",
        height=500,
        bubble_full_width=False
    )

    msg_input = gr.Textbox(
        label="输入消息",
        placeholder="请输入内容...",
        max_lines=3
    )

    # 事件绑定
    msg_input.submit(
        gradio_chat,
        inputs=[msg_input, chat_state, model_selector],
        outputs=[chatbot, msg_input],
        show_progress="hidden"
    )

    clear_btn.click(
        fn=lambda: ([], []),  # 同时清空显示和状态
        inputs=None,
        outputs=[chatbot, chat_state],
        queue=False
    )

if __name__ == "__main__":
    demo.launch(
        server_port=8001,
        share=False,
        show_error=True
    )