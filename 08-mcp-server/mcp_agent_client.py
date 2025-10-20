#!/usr/bin/env python3
"""
MCP Agent Client（LLM-driven Client）

将 MCP 服务器的工具映射为 LangChain 工具，由大模型自动选择与调用。
本脚本只负责“客户端/Agent”侧逻辑，服务端需单独启动。

运行：
  1) 先启动服务端：
     python 08-mcp-server/mcp_agent_server.py

  2) 在本目录配置 .env 或设置环境变量：OPENAI_API_KEY

  3) 再启动客户端：
     python 08-mcp-server/mcp_agent_client.py

说明：当前传输为本地子进程 stdin/stdout 的最小实现。为保持“客户端/服务端”职责清晰，客户端不自动启动服务端。
"""
import os
import sys
import json
import time
import argparse
import subprocess
from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError

try:
    from websocket import create_connection  # websocket-client
except Exception:
    create_connection = None

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor


SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server_demo.py")


class MCPTransport(ABC):
    """抽象传输层，支持不同服务端通信方式（stdio/http/ws）。"""

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...


class StdioTransport(MCPTransport):
    def __init__(self, server_path: str) -> None:
        self.server_path = server_path
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.proc and self.proc.stdout
        ready_line = self.proc.stdout.readline().strip()
        if ready_line:
            data = json.loads(ready_line)
            if not data.get("ready"):
                raise RuntimeError("Server not ready")

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        assert self.proc and self.proc.stdin and self.proc.stdout
        payload = {"id": int(time.time()), "method": method, "params": params}
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        return json.loads(line)

    def stop(self) -> None:
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


class HTTPTransport(MCPTransport):
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def start(self) -> None:
        # HTTP 传输无需额外启动
        pass

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"id": int(time.time()), "method": method, "params": params}
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")
                return json.loads(text)
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"HTTP 请求失败: {e}")

    def stop(self) -> None:
        pass


class WebSocketTransport(MCPTransport):
    def __init__(self, url: str) -> None:
        if create_connection is None:
            raise RuntimeError("缺少 websocket-client，请先安装：pip install websocket-client")
        self.url = url
        self.ws = None

    def start(self) -> None:
        self.ws = create_connection(self.url, timeout=30)

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        assert self.ws is not None
        payload = {"id": int(time.time()), "method": method, "params": params}
        self.ws.send(json.dumps(payload))
        text = self.ws.recv()
        return json.loads(text)

    def stop(self) -> None:
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass


class MCPClient:
    """与 MCP 服务器交互的客户端封装，底层可选 stdio/http/ws 传输。"""

    def __init__(self, transport: MCPTransport) -> None:
        self.transport = transport

    def start(self) -> None:
        self.transport.start()

    def list_tools(self) -> Dict[str, Any]:
        return self.transport.call("list_tools", {})

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.transport.call("call_tool", {"name": name, "args": args})

    def stop(self) -> None:
        self.transport.stop()


# ============================
# 环境加载与 LLM 配置
# ============================
def load_environment() -> None:
    """加载环境变量（支持当前目录 .env）"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


def get_llm() -> ChatOpenAI:
    """创建并配置语言模型实例"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY，请在环境或 .env 中配置")

    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": True,
        "base_url": base_url,
    }
    return ChatOpenAI(**kwargs)


def build_agent_with_mcp_tools(client: MCPClient) -> AgentExecutor:
    """将 MCP 服务器工具映射为 LangChain 工具，并构建 AgentExecutor"""

    def _extract_output(resp: Dict[str, Any]) -> Any:
        result = resp.get("result", {})
        if isinstance(result, dict) and "error" in result:
            raise ValueError(str(result.get("error")))
        return result.get("output")

    # 动态工具映射：从服务端读取工具元数据，生成 LangChain 工具
    resp = client.list_tools()
    server_tools = resp.get("result", {}).get("tools", [])

    def _tool_call(_name: str, _kwargs: Dict[str, Any]) -> Any:
        return _extract_output(client.call_tool(_name, _kwargs))

    def _python_type(t: str) -> str:
        mapping = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
        }
        return mapping.get(t, "Any")

    tools: List[Any] = []
    for meta in server_tools:
        name = meta.get("name")
        description = meta.get("description", "")
        args_schema = meta.get("args_schema", [])

        # 组装函数参数签名代码与 kwargs 构造
        params: List[str] = []
        kwargs_items: List[str] = []
        for arg in args_schema:
            arg_name = arg.get("name")
            arg_type = _python_type(str(arg.get("type", "string")))
            required = bool(arg.get("required", False))
            if required:
                params.append(f"{arg_name}: {arg_type}")
            else:
                params.append(f"{arg_name}: {arg_type} = None")
            kwargs_items.append(f"'{arg_name}': {arg_name}")

        params_code = ", ".join(params)
        kwargs_code = ", ".join(kwargs_items)
        fn_name = f"mcp_{name}"
        fn_code = (
            "def {}({}):\n"
            "    return _tool_call('{}', {{{}}})\n"
        ).format(fn_name, params_code, name, kwargs_code)

        # 在受控命名空间中创建函数
        ns: Dict[str, Any] = {"_tool_call": _tool_call, "Any": Any}
        exec(fn_code, ns)
        fn = ns[fn_name]
        fn.__doc__ = description or f"MCP 工具：{name}"

        # 使用 LangChain 的 @tool 装饰器包装为工具对象，并保持服务端工具名一致
        tool_obj = tool(name)(fn)
        tools.append(tool_obj)

    # 提示模板：明确工具使用与返回要求
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个能够使用工具的助手。根据可用 MCP 工具（由服务器动态发现）解决问题，并用中文解释你的步骤和结果。",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = get_llm()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def demo(agent_executor: AgentExecutor) -> None:
    """运行若干演示问题，展示模型自动工具调用"""
    cases = [
        "现在的时间是多少？",
        "请把『你好 MCP』原样回显出来",
        "请计算 15 加 23 的结果",
    ]
    for q in cases:
        print(f"\n> 用户：{q}")
        resp = agent_executor.invoke({"input": q})
        print(f"助手：{resp.get('output')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP Agent Client")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "ws"],
        default="stdio",
        help="选择传输层：stdio（默认）/ http / ws",
    )
    parser.add_argument(
        "--server-path",
        default=SERVER_PATH,
        help="当 transport=stdio 时使用的服务器脚本路径",
    )
    parser.add_argument(
        "--http-url",
        default="http://localhost:8000/rpc",
        help="当 transport=http 时请求的 RPC URL（POST）",
    )
    parser.add_argument(
        "--ws-url",
        default="ws://localhost:8001/rpc",
        help="当 transport=ws 时连接的 WebSocket RPC 地址",
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()

    # 根据传输层选择底层实现
    if args.transport == "stdio":
        print("[传输] stdio：将以子进程方式启动本地服务器脚本")
        transport: MCPTransport = StdioTransport(args.server_path)
    elif args.transport == "http":
        print(f"[传输] http：将连接到 {args.http_url}（请确保远端服务器已启动并支持 /rpc）")
        transport = HTTPTransport(args.http_url)
    else:
        print(f"[传输] ws：将连接到 {args.ws_url}（请确保远端服务器已启动并支持 WebSocket RPC）")
        transport = WebSocketTransport(args.ws_url)

    client = MCPClient(transport)
    client.start()
    try:
        agent_executor = build_agent_with_mcp_tools(client)
        demo(agent_executor)
    finally:
        client.stop()


if __name__ == "__main__":
    main()