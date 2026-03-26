"""
Multi-Agent Pipeline using AutoGen (pyautogen).
Agents: Planner -> Drafter -> Critic -> Editor -> FinalAnswer
"""
import autogen


# ============================================================
# System Prompts
# ============================================================

PLANNER_PROMPT = """You are the Planner agent. Your role is to:
1. Carefully read and analyze the given question.
2. Create a clear, step-by-step plan to solve it.
3. Identify what information or calculations are needed.
4. Pass the plan to the Drafter.

Be concise but thorough. Focus on the logical steps needed to arrive at the answer.
Do NOT provide the final answer yourself — only the plan."""

DRAFTER_PROMPT = """You are the Drafter agent. Your role is to:
1. Follow the plan provided by the Planner.
2. Execute each step of the plan.
3. Show your work and reasoning for each step.
4. Produce a draft answer based on the plan.

Show your calculations or reasoning clearly. Provide a clear draft answer at the end."""

CRITIC_PROMPT = """You are the Critic agent. Your role is to:
1. Review the Drafter's work carefully.
2. Check for logical errors, calculation mistakes, or misunderstandings.
3. Verify the reasoning against the original question.
4. Point out any issues or suggest improvements.

IMPORTANT: Keep your response concise. Do not repeat the same phrases or sentences.

If the answer looks correct, say "The draft answer looks correct and there are no errors. We are ready for the FinalAnswer."
If there are errors, clearly explain what is wrong and what should be corrected."""

EDITOR_PROMPT = """You are the Editor agent. Your role is to:
1. Take the Critic's feedback into account.
2. If there were errors, correct them with clear reasoning.
3. If the answer was correct, refine and polish it.
4. Produce a clean, final version of the answer.

Your output should be a corrected/refined answer with clear reasoning.
IMPORTANT: Keep your explanation concise. Do NOT repeat the same sentences or phrases over and over.

Once the solution is complete and correct without further modifications, explicitly state "The solution is complete. Pass to FinalAnswer." so the conversation can move to the final stage."""

FINAL_ANSWER_PROMPT = """You are the FinalAnswer agent. Your role is to:
1. Review the entire conversation and the Editor's refined answer.
2. Extract the FINAL answer as a short, precise value.
3. Output ONLY the final answer value, then say TERMINATE.

Your response format MUST be exactly:
FINAL ANSWER: <answer>
TERMINATE

Where <answer> is the short, precise answer (a number, a word, a phrase — whatever the question asks for).
Do NOT include any explanation — just the answer value."""


# ============================================================
# Agent Builder
# ============================================================

ORCHESTRATOR_PROMPT = """You are the Orchestrator agent. You are the dynamic coordinator of a multi-agent team solving complex math problems.
The team consists of:
- Planner: Creates the initial strategy and breaks down standard operating procedures.
- Drafter: Writes the mathematical draft solving the problem based on the plan.
- Critic: Rigorously checks for logical errors, calculation mistakes, or misunderstandings.
- Editor: Refines the answer or fixes any errors pointed out by the Critic.
- FinalAnswer: Concludes the conversation and states the final result.

Your job is to read the conversation so far, analyze the current state of problem-solving, and logically decide who should be the NEXT agent to speak to move the team towards a correct answer.

Guidelines for Dynamic Routing:
- You have full authority to alter the flow based on context. 
- If the Planner's strategy seems fundamentally flawed, you can route back to Planner.
- If the Drafter provides a solid answer, you should usually send it to the Critic to verify.
- If the Critic finds errors, send it to the Editor to fix them.
- If the Critic or Editor confirm the answer is completely correct and verified, route immediately to FinalAnswer.
- If an agent is stuck in a loop saying the same thing or the overall chat has gone more than 10 turns, forcefully route to FinalAnswer to conclude.

Briefly explain your reasoning (in 1-2 lines), and then cleanly pick the NEXT agent to speak. 
At the very end of your response, you MUST output the exact name of the next agent in the format: "NEXT: [AgentName]".
Example:
"The Drafter has provided a solution, but the Critic found a calculation error. We need the Editor to fix this.
NEXT: Editor"
"""

def create_agents(llm_config):
    """Create the multi-agent team and return agents."""

    is_term = lambda x: x.get("content", "") and "TERMINATE" in x.get("content", "")

    planner = autogen.AssistantAgent(
        name="Planner",
        description="Creates the initial plan.",
        system_message=PLANNER_PROMPT,
        llm_config=llm_config,
        is_termination_msg=is_term,
    )

    drafter = autogen.AssistantAgent(
        name="Drafter",
        description="Drafts the solution.",
        system_message=DRAFTER_PROMPT,
        llm_config=llm_config,
        is_termination_msg=is_term,
    )

    critic = autogen.AssistantAgent(
        name="Critic",
        description="Reviews the solution.",
        system_message=CRITIC_PROMPT,
        llm_config=llm_config,
        is_termination_msg=is_term,
    )

    editor = autogen.AssistantAgent(
        name="Editor",
        description="Edits the solution.",
        system_message=EDITOR_PROMPT,
        llm_config=llm_config,
        is_termination_msg=is_term,
    )

    final_answer = autogen.AssistantAgent(
        name="FinalAnswer",
        description="Terminates the conversation.",
        system_message=FINAL_ANSWER_PROMPT,
        llm_config=llm_config,
        is_termination_msg=is_term,
    )
    
    orchestrator = autogen.AssistantAgent(
        name="Orchestrator",
        description="Dynamically evaluates the conversation and decides who speaks next.",
        system_message=ORCHESTRATOR_PROMPT,
        llm_config=llm_config,
    )

    return [planner, drafter, critic, editor, final_answer, orchestrator]


def speaker_selection_func(last_speaker, group_chat):
    """
    Custom speaker selection that uses the Orchestrator to decide, but executes the Orchestrator's choice.
    """
    agents = group_chat.agents
    agent_map = {a.name: a for a in agents}
    
    # 항상 Planner가 먼저 시작합니다.
    if last_speaker.name == "UserProxy":
        return agent_map["Planner"]
        
    # Orchestrator가 아닌 일반 에이전트가 말하고 나면, 무조건 Orchestrator를 호출해 다음 타자를 정하게 합니다.
    if last_speaker.name != "Orchestrator" and last_speaker.name != "FinalAnswer":
        return agent_map["Orchestrator"]
        
    # Orchestrator가 방금 발언했다면, 남긴 문자열에서 다음 타자를 파싱합니다.
    if last_speaker.name == "Orchestrator":
        messages = group_chat.messages
        if not messages:
            return agent_map["FinalAnswer"]
            
        last_msg = messages[-1].get("content", "")
        # "NEXT: Critic" 형태의 텍스트가 있는지 찾습니다.
        for name in ["Planner", "Drafter", "Critic", "Editor", "FinalAnswer"]:
            if f"NEXT: {name}" in last_msg or f"NEXT: {name}" in last_msg.replace(" ", ""):
                return agent_map[name]
                
        # 기본값
        return agent_map["Critic"]

    return None

def build_group_chat(llm_config, max_round=15):
    """Build GroupChat with all agents and a UserProxy."""

    agents = create_agents(llm_config)

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        description="문제를 제시하는 역할을 하며, 다른 에이전트들이 문제를 해결하도록 돕습니다.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
        is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x.get("content", ""),
    )

    all_agents = [user_proxy] + agents

    # 명시적 라우터 방식을 사용하므로 speaker_selection_func 지정
    group_chat = autogen.GroupChat(
        agents=all_agents,
        messages=[],
        max_round=max_round,
        speaker_selection_method=speaker_selection_func,
        allow_repeat_speaker=False,
    )

    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    return user_proxy, group_chat, manager
