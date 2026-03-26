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

If the answer looks correct, say "The draft answer looks correct" and briefly explain why.
If there are errors, clearly explain what is wrong and what should be corrected."""

EDITOR_PROMPT = """You are the Editor agent. Your role is to:
1. Take the Critic's feedback into account.
2. If there were errors, correct them with clear reasoning.
3. If the answer was correct, refine and polish it.
4. Produce a clean, final version of the answer.

Your output should be a corrected/refined answer with clear reasoning."""

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

def create_agents(llm_config):
    """Create the multi-agent team and return (agents, group_chat, manager)."""

    planner = autogen.AssistantAgent(
        name="Planner",
        system_message=PLANNER_PROMPT,
        llm_config=llm_config,
    )

    drafter = autogen.AssistantAgent(
        name="Drafter",
        system_message=DRAFTER_PROMPT,
        llm_config=llm_config,
    )

    critic = autogen.AssistantAgent(
        name="Critic",
        system_message=CRITIC_PROMPT,
        llm_config=llm_config,
    )

    editor = autogen.AssistantAgent(
        name="Editor",
        system_message=EDITOR_PROMPT,
        llm_config=llm_config,
    )

    final_answer = autogen.AssistantAgent(
        name="FinalAnswer",
        system_message=FINAL_ANSWER_PROMPT,
        llm_config=llm_config,
    )

    return [planner, drafter, critic, editor, final_answer]


def speaker_selection_func(last_speaker, group_chat):
    """
    Custom speaker selection to enforce the flow:
    UserProxy -> Planner -> Drafter -> Critic -> Editor -> FinalAnswer
    With optional Debate loop: if Critic finds errors, go back to Editor.
    """
    agents = group_chat.agents
    agent_map = {a.name: a for a in agents}

    if last_speaker.name == "UserProxy":
        return agent_map["Planner"]
    elif last_speaker.name == "Planner":
        return agent_map["Drafter"]
    elif last_speaker.name == "Drafter":
        return agent_map["Critic"]
    elif last_speaker.name == "Critic":
        return agent_map["Editor"]
    elif last_speaker.name == "Editor":
        # Check message count to decide if we should loop or finalize
        messages = group_chat.messages
        # Count how many times Editor has spoken
        editor_count = sum(1 for m in messages if m.get("name") == "Editor")
        if editor_count >= 2:
            # After 2 rounds of editing, finalize
            return agent_map["FinalAnswer"]
        else:
            # Allow one more review cycle
            return agent_map["Critic"]
    elif last_speaker.name == "FinalAnswer":
        return None  # End
    else:
        return agent_map["Planner"]


def build_group_chat(llm_config, max_round=12):
    """Build GroupChat with all agents and a UserProxy."""

    agents = create_agents(llm_config)

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    all_agents = [user_proxy] + agents

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
