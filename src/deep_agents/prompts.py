# prompts for deep agents
clarify_with_user="""
以下是用户请求报告时至今为止发送的消息:
<Messages>
{messages}
</Messages>

Today's date is {date}.

评估您是否需要提出澄清问题, 或者用户是否已提供足够的信息供您开始研究
IMPORTANT: 如果您在消息历史记录中看到您已经提出过澄清问题，则几乎总是不需要再次提问。仅在绝对必要时才提出另一个问题

如果出现缩写、简称或未知术语，请用户进行澄清

如果您需要提问，请遵循以下准则：
- 收集所有必要信息时力求简洁
- 确保以简洁、结构化的方式收集完成研究任务所需的所有信息
- 如果合适，请使用项目符号或编号列表以使其更清晰。请确保使用 Markdown 格式，并且能够被阅读。如果字符串输出被传递给 Markdown 渲染器，则渲染正确
- 不要询问不必要的信息，或用户已经提供的信息。如果您发现用户已经提供了信息，请不要再次询问

请使用以下键以有效的 JSON 格式进行响应：
"need_clarification": boolean,
"question": "<question to ask the user to clarify the report scope>",
"verification": "<verification message that we will start research>"

如果需要询问澄清问题，请返回：
"need_clarification": true,
"question": "<your clarifying question>",
"verification": ""

如果不需要询问澄清问题，请返回：
"need_clarification": false,
"question": "",
"verification": "<acknowledgement message that you will now start research based on the provided information>"

对于不需要澄清时的确认信息：
- 确认您已掌握足够的信息继续
- 简要总结您从对方请求中理解的关键要点
- 确认您将开始研究工作
- 保持信息简洁专业
""".strip()

research_brief ="""
你将收到一组你与用户迄今为止的消息, 你的任务是讲这些消息转化成一个清晰的研究主题列表, 以指导你的研究工作. 你应该分析消息内容, 提取出用户的主要兴趣点和需求, 并将它们组织成一个结构化的研究主题列表
<Messages>
{messages}
</Messages>

今天的日期是 {current_date}.

你需要返回一个研究问题, 以指导后续研究
指导原则:

1. 尽可能具体和详细

- 包含所有已知的用户偏好，并明确列出需要考虑的关键属性或维度。

- 务必将用户提供的所有详细信息包含在说明中。

2. 将未明确说明但必要的维度填写为开放式问题

- 如果某些属性对于有意义的输出至关重要，但用户未提供，请明确说明这些属性是开放式的，或者默认为无特定约束。

3. 避免不合理的假设

- 如果用户没有提供特定细节，请不要凭空捏造。

- 相反，应说明信息缺失，并引导研究人员灵活处理或接受所有可能的选项。

4. 使用第一人称

- 从用户的角度提出请求。

5. 信息来源
- 如果需要优先考虑特定信息来源，请在研究问题中明确指出
- 如果没有特定语言或地区偏好，请说明可以使用任何相关来源
""".strip()

planner="""
Research Topcic:
{research_topic}

请为该问题生成研究大纲喝研究假设，以指导后续的研究工作

请使用以下键以有效的 JSON 格式进行响应：

"hypothesis_1": "关于市场/行业趋势的假设（需要验证）",
"hypothesis_2": "关于竞争格局或技术发展的假设（需要验证）",
"hypothesis_3": "关于政策或外部因素影响的假设（需要验证）",
"sec_1_title": "市场概况",
"sec_1_desc": "描述市场规模、增速",
"sec_1_query": "搜索关键词",
"sec_2_title": "竞争格局",
"sec_2_desc": "描述主要企业",
"sec_2_query": "搜索关键词",
"sec_3_title": "技术趋势",
"sec_3_desc": "描述核心技术",
"sec_3_query": "搜索关键词",
"sec_4_title": "政策环境",
"sec_4_desc": "描述相关政策",
"sec_4_query": "搜索关键词",
"sec_5_title": "挑战机遇",
"sec_5_desc": "描述挑战和机会",
"sec_5_query": "搜索关键词",
"sec_6_title": "未来展望",
"sec_6_desc": "描述发展趋势",
"sec_6_query": "搜索关键词",
"questions": "核心问题1;核心问题2;核心问题3"


研究假设示例：
- 假设市场规模将持续增长，需要用数据验证增速
- 假设某类技术会成为主流，需要找证据支持或反驳
- 假设政策变化会影响行业格局，需要分析政策走向

请根据研究课题填写具体内容，每个字段都是字符串类型。

"""


REVISION_PROMPT = """你是总架构师，需要根据研究进展动态调整大纲。

## 原始问题
{query}

## 当前大纲
{current_outline}

## 新发现的重要信息
{new_findings}

## 当前进度
- 已完成章节: {completed_sections}
- 收集的事实数量: {facts_count}
- 发现的数据点: {data_points_count}

## 任务
评估是否需要调整大纲。可能的调整包括：
1. 新增章节（发现了重要的新方向）
2. 删除章节（发现某方向信息太少）
3. 调整章节顺序或优先级
4. 细化或合并章节

请使用以下键以有效的 JSON 格式进行响应：
"needs_revision": boolean,
"revision_reason": "调整原因",
"revised_outline": [...],  // 如果needs_revision为true
"new_search_queries": ["新增的搜索关键词"]  // 如果需要补充搜索

"""

