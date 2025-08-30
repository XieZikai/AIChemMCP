# agent.py
import subprocess
import json
import sys
import time
from threading import Thread
from llm_client import OpenAI_LLM


class Agent:
    def __init__(self):
        self.servers = {}
        self.tools = {}
        self.history = []
        self.llm_client = OpenAI_LLM()

    def start_server(self, name: str, command: list):
        """启动一个服务器子进程并管理其IO管道"""
        print(f"[AGENT] Starting server: {name}...")
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
            universal_newlines=True
        )
        self.servers[name] = {
            "process": process,
            "stdin": process.stdin,
            "stdout": process.stdout,
        }
        # 启动一个线程来打印服务器的stderr日志，以免阻塞
        Thread(target=self._log_stderr, args=(name, process.stderr), daemon=True).start()
        print(f"[AGENT] Server '{name}' started.")

    def _log_stderr(self, name, stderr_pipe):
        """在后台打印服务器的日志"""
        for line in iter(stderr_pipe.readline, ''):
            print(f"[{name} LOG] {line.strip()}", file=sys.stderr)

    def discover_tools(self):
        """从所有启动的服务器接收advertise消息，并构建工具列表"""
        print("\n[AGENT] Discovering tools from all servers...")
        for name, server in self.servers.items():
            # MCP规定，服务器启动后第一条消息必须是advertise
            adv_line = server['stdout'].readline()
            adv_data = json.loads(adv_line)

            server_tools = adv_data.get("params", {}).get("server", {}).get("capabilities", {}).get("tools", [])
            for tool in server_tools:
                tool_name = tool['name']
                self.tools[tool_name] = tool
                self.tools[tool_name]['server_name'] = name  # 记录这个工具属于哪个服务器
            print(f"[AGENT] Discovered {len(server_tools)} tools from '{name}'.")
        print(f"[AGENT] Tool discovery complete. Total tools available: {len(self.tools)}")

    def build_system_prompt(self) -> str:
        """构建包含角色、工具列表和“脚手架”的系统提示"""
        prompt = "# 角色与总目标\n"
        prompt += "你是一位顶级的AI研究化学家。你的任务是利用实验室的自动化工具，高效、安全地完成科研目标。\n\n"

        prompt += "# 可用工具\n---\n"
        for name, tool in self.tools.items():
            prompt += f"- 工具: {name}\n"
            prompt += f"  - 描述: {tool['description']}\n"
            prompt += f"  - 参数: {json.dumps(tool['parameters'])}\n"
        prompt += "---\n\n"

        prompt += "# 核心工作流提示 (SOPs & Best Practices)\n"
        prompt += "当你需要执行一个**优化任务**时，一个高效、标准的流程是“贝叶斯优化循环”。强烈建议你遵循以下步骤：\n"
        prompt += "1.  **初始化 (Initialize)**: (如果需要) 调用 `initialize` 来开始一个新的优化会话。\n"
        prompt += "2.  **建议 (Suggest)**: 调用 `suggest` 工具，获取下一步的实验参数。\n"
        prompt += "3.  **执行 (Execute)**: 使用 `robotic_reaction` 和 `robotic_measurement` 工具来运行实验并测量结果。\n"
        prompt += "4.  **观察 (Observe)**: 调用 `observe` 工具，将新的实验参数和结果反馈给优化器。\n"
        prompt += "5.  **重复 (Repeat)**: 持续这个循环，直到达成优化目标。\n"

        return prompt

    def dispatch_tool_call(self, tool_call: dict) -> dict:
        """将工具调用请求分发给正确的服务器并返回结果"""
        method = tool_call.get("method")
        if not method or method not in self.tools:
            return {"error": f"Tool '{method}' not found."}

        # 根据工具名找到对应的服务器
        server_name = self.tools[method]['server_name']
        server = self.servers[server_name]

        # 构建并发送JSON-RPC请求
        request_id = int(time.time() * 1000)
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": tool_call.get("params", {}),
            "id": request_id
        }

        print(f"[AGENT] -> [{server_name}] {json.dumps(request)}")
        server['stdin'].write(json.dumps(request) + '\n')
        server['stdin'].flush()

        # 等待并读取响应
        while True:
            response_line = server['stdout'].readline()
            if not response_line:
                return {"error": "Server connection closed."}
            response = json.loads(response_line)
            if response.get("id") == request_id:
                print(f"[AGENT] <- [{server_name}] {json.dumps(response)}")
                return response

    def build_planner_system_prompt(self) -> str:
        """为规划阶段构建专用的系统提示"""
        prompt = "# 角色与目标\n"
        prompt += "你是一个专家级的实验室工作流规划师。你的任务是根据用户的目标，将一系列可用的基础工具，组合成一个逻辑正确的、有序的步骤列表来完成任务。\n\n"

        prompt += "# 可用工具\n---\n"
        for name, tool in self.tools.items():
            # 只提供名称和描述，让LLM专注于逻辑，而非实现细节
            prompt += f"- {name}: {tool['description']}\n"
        prompt += "---\n\n"

        prompt += "# 任务\n"
        prompt += "请分析用户的目标，并生成一个包含所有必要步骤的完整计划。你的最终输出必须是一个调用`submit_workflow_plan`工具的请求，其中包含所有步骤。\n"
        return prompt

    def plan_workflow(self, user_goal: str) -> list:
        """【规划阶段】调用LLM生成一个多步计划"""
        print("\n[AGENT] Entering PLANNING stage...")
        system_prompt = self.build_planner_system_prompt()
        plan = self.llm_client.generate_plan(system_prompt, user_goal, self.tools)

        print("[AGENT] Plan generated successfully.")
        # for i, step in enumerate(plan, 1):
        #     print(f"  Step {i}: {step['method']}({json.dumps(step['params'])})")
        return plan

    def run(self):
        """启动主交互循环"""
        system_prompt = self.build_system_prompt()
        self.history.append({"role": "system", "content": system_prompt})

        print("\n--- Agent is Ready ---")
        print("System Prompt has been constructed. You can now interact with the agent.")
        print("Try typing: 'Please optimize the reaction yield.'")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                for server in self.servers.values():
                    server['process'].terminate()
                print("[AGENT] All servers terminated. Exiting.")
                break

            self.history.append({"role": "user", "content": user_input})

            print("[AGENT] Thinking with OpenAI...")
            # 将system_prompt, history, 和 self.tools 全部传给决策函数
            llm_response = self.llm_client.get_decision(
                system_prompt,
                self.history,
                self.tools
            )
            print(f"[LLM] Thought: {llm_response.get('thought')}")
            self.history.append({"role": "assistant", "content": llm_response})

            # 2. 如果LLM决定调用工具，则分发
            if "tool_call" in llm_response:
                tool_call = llm_response['tool_call']
                print(f"[AGENT] Dispatching tool call: {tool_call.get('method')}")

                result = self.dispatch_tool_call(tool_call)
                self.history.append({"role": "tool_result", "content": result})
                print(f"[AGENT] Tool execution result: {result}")

                # 模拟工具执行后的再次思考
                print("[AGENT] Thinking about the result...")
                next_llm_response = self.llm_client.get_decision(system_prompt, self.history, self.tools)
                print(f"[LLM] Next Thought: {next_llm_response.get('thought')}")
                if "speak" in next_llm_response:
                    print(f"\nAgent: {next_llm_response['speak']}")
            elif "speak" in llm_response:
                print(f"\nAgent: {llm_response['speak']}")


if __name__ == "__main__":
    agent = Agent()

    # 启动所有需要的服务器
    agent.start_server("BO_Server", ["python", "dummy_planner_servers/dummy_bo_server.py"])

    # 等待服务器启动并完成工具发现
    time.sleep(1)  # 给予服务器一点启动时间
    agent.discover_tools()

    # 启动Agent的主循环
    # agent.run()

    # 测试Agent的plan_workflow函数
    user_goal = "我想要规划一个贝叶斯优化工作流。这个工作流需要包含三个循环。请为我规划这个流程。这个工作流应该使用已经储存的数据作为启动。"
    plan = agent.plan_workflow(user_goal)
    print(plan)
