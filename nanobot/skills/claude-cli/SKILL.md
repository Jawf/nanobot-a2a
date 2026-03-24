---
name: claude-cli
description: "通过聊天渠道调用 Claude CLI（claude-code）进行多轮对话。支持代码任务委托、AI 编程助手交互，并将 Claude 的完整回复友好地推送回聊天窗口。"
metadata: {"nanobot":{"emoji":"🤖","requires":{"bins":["claude"]},"always":false}}
---

# Claude CLI 对话 Skill

本 Skill 让用户可以在任意聊天渠道（飞书、Telegram、Slack 等）中直接与 **Claude CLI（claude-code）** 进行对话，支持多轮连续会话，Claude 的回复会完整推送回聊天界面。

---

## 何时使用

当用户提出以下任一需求时，使用本 Skill：

- "帮我用 Claude CLI 问一下…"
- "把这个需求发给 claude-code，让它帮我…"
- "开一个 Claude CLI 会话"
- "用 claude 分析 / 修复 / 生成代码…"
- "和 claude 聊一下这个问题"
- "继续上次和 Claude 的对话"
- "重新开一个新的 Claude 会话"

---

## 可用工具

### `claude_cli`

调用本地安装的 `claude` CLI，将提示词发送给 Claude Code 并将回复流式推送到当前聊天渠道。

**参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | ✅ | 发给 Claude CLI 的提示词或任务描述 |
| `new_session` | boolean | ❌ | `true` = 开启全新会话（默认 `false`，继续上次对话） |
| `timeout` | integer | ❌ | 等待超时秒数（默认 300，最大 600） |

**多轮会话机制：**
- 每个聊天窗口（channel + chat_id）独立维护一个 Claude CLI 会话 ID
- 默认情况下，同一聊天的后续调用会自动 `--resume` 上次会话
- 设置 `new_session: true` 可清除会话历史、重新开始

---

## 执行流程

1. **识别意图**：判断用户是想发新消息给 Claude、继续会话还是重置会话。
2. **提取 prompt**：将用户需求整理为清晰的提示词（不要无谓缩短，保留细节）。
3. **调用工具**：调用 `claude_cli` 工具，传入 prompt 和可选参数。
4. **回复用户**：Claude CLI 的回复会自动推送到聊天渠道，无需再次转发。告知用户结果是否成功，如有错误简要说明原因。

---

## 交互示例

### 示例 1：单次提问

**用户**：帮我用 claude 解释一下什么是依赖注入

**执行**：
```
claude_cli(prompt="请解释什么是依赖注入，用 Python 举例说明")
```

**输出**：Claude 的回答直接推送到聊天界面。

---

### 示例 2：多轮对话

**用户**：（接上）再给我看一个用 FastAPI 的例子

**执行**（自动续接上次会话）：
```
claude_cli(prompt="再给我看一个使用 FastAPI 框架的依赖注入示例")
```

---

### 示例 3：开新会话

**用户**：重新开一个 Claude 会话，问一下 TypeScript 的泛型

**执行**：
```
claude_cli(
  prompt="请介绍 TypeScript 中的泛型，并给出常用场景示例",
  new_session=True
)
```

---

### 示例 4：代码任务委托

**用户**：让 claude 帮我写一个 Python 函数，解析 JSONL 文件并过滤空行

**执行**：
```
claude_cli(
  prompt="请写一个 Python 函数，功能：读取 JSONL 文件，跳过空行和解析失败的行，返回所有成功解析的对象列表。要求有类型注解和简短文档字符串。",
  timeout=120
)
```

---

## 注意事项

- **需要本地安装 `claude` CLI**：通过 `npm install -g @anthropic-ai/claude-code` 安装，并确认 `claude` 在 PATH 中。
- **会话隔离**：不同聊天窗口的 Claude 会话相互独立，不会串话。
- **长任务**：对于涉及文件操作或复杂分析的任务，可适当增加 `timeout`（最大 600s）。
- **响应推送**：Claude 的完整回复会直接推送到聊天窗口，不需要再次复述内容，只需告知用户是否成功。
