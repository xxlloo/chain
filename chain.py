import os
import requests
from dotenv import load_dotenv
import gradio as gr
import time

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")

# 模型配置（注意模型ID格式）
MODELS = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "glm": "THUDM/GLM-Z1-9B-0414"  # 根据API实际名称调整
}


def chat_with_model(model_key, messages):
    """与指定模型进行交互（带调试输出）"""
    print(f"[DEBUG] 正在使用模型: {MODELS[model_key]}")
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
    except requests.exceptions.HTTPError as e:
        print(f"API错误: {e.response.status_code} {e.response.text}")
        return "模型服务请求失败，请检查模型配置"
    except Exception as e:
        print(f"请求异常: {str(e)}")
        return "服务暂时不可用，请稍后再试。"


def gradio_chat(message, chat_history, model_key):
    """处理聊天交互（优化历史记录处理）"""
    try:
        # 构建消息队列（包含历史）
        messages = []
        for entry in chat_history:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                messages.append({"role": "user", "content": str(entry[0])})
                messages.append({"role": "assistant", "content": str(entry[1])})
        messages.append({"role": "user", "content": str(message)})

        # 获取响应
        full_response = chat_with_model(model_key, messages)

        # 流式输出模拟
        chat_history.append((message, ""))
        for char in full_response:
            chat_history[-1] = (chat_history[-1][0], chat_history[-1][1] + char)
            time.sleep(0.015)
            yield chat_history, ""
        # 回复结束后，返回特殊标志用于前端JS关闭计时器
        yield gr.update(), "__STOP_TIMER__"
    except Exception as e:
        print(f"对话处理异常: {str(e)}")
        yield chat_history, "系统出现异常，请稍后再试"


# 创建界面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 多模型对话系统（实时切换版）")

    # 状态管理
    chat_state = gr.State([])

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=[
                ("DeepSeek-7B", "deepseek"),
                ("GLM-9B", "glm")
            ],
            value="deepseek",
            label="选择模型",
            interactive=True
        )
        clear_btn = gr.Button("清空对话", variant="stop")

    chatbot = gr.Chatbot(
        label="对话历史",
        height=500,
        bubble_full_width=False,
        avatar_images=(
            "https://cdn.oaistatic.com/assets/favicon-miwirzcw.ico",
            "https://cdn.oaistatic.com/assets/favicon-miwirzcw.ico"
        )
    )

    msg_input = gr.Textbox(
        label="输入消息",
        placeholder="请输入内容...",
        max_lines=3,
        container=False
    )

    # 右下角计时器
    timer_html = gr.HTML(
        """
        <div id="timer-box" style="position:fixed;right:30px;bottom:30px;z-index:9999;display:none;font-size:18px;background:#fff3;border-radius:8px;padding:8px 16px;box-shadow:0 2px 8px #0002;">
            ⏱️ <span id="timer-val">0.00</span> 秒
        </div>
        <script>
        window.timerInterval = null;
        window.timerStart = null;
        function startTimer() {
            document.getElementById('timer-box').style.display = 'block';
            window.timerStart = Date.now();
            window.timerInterval = setInterval(function() {
                let elapsed = (Date.now() - window.timerStart) / 1000;
                document.getElementById('timer-val').innerText = elapsed.toFixed(2);
            }, 30);
        }
        function stopTimer() {
            if(window.timerInterval) clearInterval(window.timerInterval);
            document.getElementById('timer-box').style.display = 'none';
            document.getElementById('timer-val').innerText = '0.00';
        }
        // 监听输入框提交，启动计时器
        function bindInputTimer() {
            let input = document.querySelector('textarea');
            if(input) {
                input.addEventListener('keydown', function(e){
                    if(e.key === 'Enter' && !e.shiftKey){
                        setTimeout(startTimer, 100); // 稍微延迟，防止多次触发
                    }
                });
            }
        }
        // 监听Gradio输出，遇到特殊标志关闭计时器
        function observeStopTimer() {
            let obs = new MutationObserver(function(muts){
                muts.forEach(function(mut){
                    if(mut.addedNodes) {
                        mut.addedNodes.forEach(function(node){
                            if(node.innerText && node.innerText.includes("__STOP_TIMER__")) {
                                stopTimer();
                            }
                        });
                    }
                });
            });
            let root = document.body;
            obs.observe(root, {childList:true, subtree:true});
        }
        window.addEventListener('DOMContentLoaded', function(){
            bindInputTimer();
            observeStopTimer();
        });
        </script>
        """,
        show_label=False
    )

    # 事件绑定
    msg_input.submit(
        gradio_chat,
        inputs=[msg_input, chat_state, model_selector],
        outputs=[chatbot, msg_input],
        show_progress="hidden"
    )

    clear_btn.click(
        fn=lambda: ([], []),
        inputs=None,
        outputs=[chatbot, chat_state],
        queue=False
    )

    # 添加模型切换监听
    model_selector.change(
        fn=lambda: ([], []),
        inputs=None,
        outputs=[chatbot, chat_state],
        queue=False
    )

if __name__ == "__main__":
    demo.launch(
        server_port=8001,
        share=False,
        show_error=True,
        debug=True  # 启用调试模式
    )