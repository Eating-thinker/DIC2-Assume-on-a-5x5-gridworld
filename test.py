import streamlit as st
from math import inf

# ==============================
# GridWorld + Value Iteration
# ==============================
GRID_SIZE = 5
DEFAULT_START = (0, 0)
DEFAULT_GOAL = (4, 4)
DEFAULT_BLOCKS = {(1, 1), (2, 2), (3, 3)}
ACTIONS = {
    "↑": (-1, 0),
    "↓": (1, 0),
    "←": (0, -1),
    "→": (0, 1),
}


def init_state():
    if "start" not in st.session_state:
        st.session_state.start = DEFAULT_START
    if "goal" not in st.session_state:
        st.session_state.goal = DEFAULT_GOAL
    if "blocks" not in st.session_state:
        st.session_state.blocks = set(DEFAULT_BLOCKS)
    if "mode" not in st.session_state:
        st.session_state.mode = "block"
    if "value_table" not in st.session_state:
        st.session_state["value_table"] = None
    if "policy" not in st.session_state:
        st.session_state.policy = None
    if "path" not in st.session_state:
        st.session_state.path = []


def reset_default():
    st.session_state.start = DEFAULT_START
    st.session_state.goal = DEFAULT_GOAL
    st.session_state.blocks = set(DEFAULT_BLOCKS)
    st.session_state["value_table"] = None
    st.session_state.policy = None
    st.session_state.path = []


def in_bounds(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE


def get_next_state(state, action, blocks):
    r, c = state
    dr, dc = action
    nr, nc = r + dr, c + dc

    if not in_bounds(nr, nc) or (nr, nc) in blocks:
        return state
    return (nr, nc)


def value_iteration(start, goal, blocks, gamma=0.9, step_reward=-1.0, goal_reward=10.0, theta=1e-6, max_iterations=1000):
    states = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if (r, c) not in blocks]
    V = {s: 0.0 for s in states}
    policy = {}

    for _ in range(max_iterations):
        delta = 0.0
        new_V = V.copy()

        for s in states:
            if s == goal:
                new_V[s] = 0.0
                continue

            action_values = []
            for arrow, move in ACTIONS.items():
                s_next = get_next_state(s, move, blocks)
                reward = goal_reward if s_next == goal else step_reward
                q = reward + gamma * V[s_next]
                action_values.append((q, arrow))

            best_q = max(q for q, _ in action_values)
            new_V[s] = best_q
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < theta:
            break

    for s in states:
        if s == goal:
            policy[s] = "G"
            continue

        best_action = None
        best_q = -inf
        for arrow, move in ACTIONS.items():
            s_next = get_next_state(s, move, blocks)
            reward = goal_reward if s_next == goal else step_reward
            q = reward + gamma * V[s_next]
            if q > best_q:
                best_q = q
                best_action = arrow
        policy[s] = best_action

    path = extract_path(start, goal, policy, blocks)
    return V, policy, path


def extract_path(start, goal, policy, blocks, max_steps=100):
    if start in blocks or goal in blocks:
        return []

    current = start
    path = [current]
    visited = {current}

    for _ in range(max_steps):
        if current == goal:
            return path

        action_arrow = policy.get(current)
        if action_arrow not in ACTIONS:
            return path

        next_state = get_next_state(current, ACTIONS[action_arrow], blocks)

        if next_state == current:
            return path

        path.append(next_state)
        current = next_state

        if current in visited and current != goal:
            return path
        visited.add(current)

    return path


def cell_clicked(r, c):
    pos = (r, c)
    mode = st.session_state.mode

    if mode == "start":
        if pos != st.session_state.goal and pos not in st.session_state.blocks:
            st.session_state.start = pos
    elif mode == "goal":
        if pos != st.session_state.start and pos not in st.session_state.blocks:
            st.session_state.goal = pos
    elif mode == "block":
        if pos != st.session_state.start and pos != st.session_state.goal:
            if pos in st.session_state.blocks:
                st.session_state.blocks.remove(pos)
            else:
                st.session_state.blocks.add(pos)

    st.session_state["value_table"] = None
    st.session_state.policy = None
    st.session_state.path = []


def build_cell_html(r, c, values, policy, path, show_values=True, show_policy=True):
    pos = (r, c)
    start = st.session_state.start
    goal = st.session_state.goal
    blocks = st.session_state.blocks

    bg = "#f7f7f7"
    border = "#999"
    title = ""
    subtitle = ""

    if pos in path and pos not in {start, goal}:
        bg = "#dbeafe"

    if pos in blocks:
        bg = "#4b5563"
        border = "#374151"
        title = "BLOCK"
        subtitle = ""
    elif pos == start:
        bg = "#22c55e"
        border = "#15803d"
        title = "START"
    elif pos == goal:
        bg = "#ef4444"
        border = "#b91c1c"
        title = "GOAL"

    if pos not in blocks and pos != start and pos != goal:
        if show_policy and policy is not None:
            title = policy.get(pos, "")
        if show_values and values is not None:
            subtitle = f"V={values.get(pos, 0.0):.2f}"

    if pos == start and show_values and values is not None:
        subtitle = f"V={values.get(pos, 0.0):.2f}"
    if pos == goal and show_values and values is not None:
        subtitle = f"V={values.get(pos, 0.0):.2f}"

    html = f"""
    <div style="
        height:90px;
        border:2px solid {border};
        border-radius:12px;
        background:{bg};
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        font-weight:700;
        color:{'white' if pos in blocks or pos in {start, goal} else '#111827'};
        text-align:center;
        padding:4px;
    ">
        <div style="font-size:24px; line-height:1.1;">{title}</div>
        <div style="font-size:12px; margin-top:6px;">({r},{c})</div>
        <div style="font-size:12px; margin-top:2px;">{subtitle}</div>
    </div>
    """
    return html


def render_grid(values, policy, path, show_values=True, show_policy=True):
    st.markdown("### Interactive Grid")
    for r in range(GRID_SIZE):
        cols = st.columns(GRID_SIZE)
        for c in range(GRID_SIZE):
            with cols[c]:
                st.markdown(
                    build_cell_html(r, c, values, policy, path, show_values, show_policy),
                    unsafe_allow_html=True,
                )
                st.button(
                    f"Select {r},{c}",
                    key=f"btn_{r}_{c}",
                    on_click=cell_clicked,
                    args=(r, c),
                    use_container_width=True,
                )


def render_policy_text(policy):
    st.markdown("### Best Policy (Text View)")
    rows = []
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            pos = (r, c)
            if pos in st.session_state.blocks:
                row.append("■")
            elif pos == st.session_state.start:
                row.append("S")
            elif pos == st.session_state.goal:
                row.append("G")
            else:
                row.append(policy.get(pos, ".") if policy else ".")
        rows.append("  ".join(row))
    st.code("\n".join(rows))


def main():
    st.set_page_config(page_title="5x5 GridWorld - Value Iteration", layout="wide")
    init_state()

    st.title("5x5 GridWorld Value Iteration Demo")
    st.write(
        "Use Streamlit to interactively set Start / Goal / Blocks, then run Value Iteration to compute the value function and optimal policy."
    )

    with st.sidebar:
        st.header("Controls")
        st.radio(
            "Edit Mode",
            ["start", "goal", "block"],
            key="mode",
            format_func=lambda x: {"start": "Set Start", "goal": "Set Goal", "block": "Toggle Block"}[x],
        )

        gamma = st.slider("Discount Factor γ", 0.1, 0.99, 0.9, 0.01)
        step_reward = st.slider("Step Reward", -5.0, 0.0, -1.0, 0.1)
        goal_reward = st.slider("Goal Reward", 1.0, 50.0, 10.0, 1.0)
        theta = st.number_input("Convergence Threshold θ", min_value=0.000001, max_value=0.1, value=0.000001, format="%.6f")

        show_values = st.checkbox("Show Value Function", value=True)
        show_policy = st.checkbox("Show Policy Arrows", value=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Run Value Iteration", use_container_width=True):
                V, P, path = value_iteration(
                    st.session_state.start,
                    st.session_state.goal,
                    st.session_state.blocks,
                    gamma=gamma,
                    step_reward=step_reward,
                    goal_reward=goal_reward,
                    theta=theta,
                )
                st.session_state["value_table"] = V
                st.session_state.policy = P
                st.session_state.path = path
        with c2:
            if st.button("Reset Default", use_container_width=True):
                reset_default()
                st.rerun()

        st.markdown("---")
        st.markdown("**Assignment default setting**")
        st.write("Start: (0,0)")
        st.write("Goal: (4,4)")
        st.write("Blocks: (1,1), (2,2), (3,3)")

    values = st.session_state["value_table"]
    policy = st.session_state.policy
    path = st.session_state.path

    left, right = st.columns([2, 1])

    with left:
        render_grid(values, policy, path, show_values=show_values, show_policy=show_policy)

    with right:
        st.markdown("### Current Setting")
        st.write(f"**Start:** {st.session_state.start}")
        st.write(f"**Goal:** {st.session_state.goal}")
        st.write(f"**Blocks:** {sorted(list(st.session_state.blocks))}")

        if values is not None and policy is not None:
            st.success("Value Iteration completed.")
            render_policy_text(policy)
            st.markdown("### Path from Start to Goal")
            if path and path[-1] == st.session_state.goal:
                st.write(" → ".join(map(str, path)))
            else:
                st.warning("No complete path found under the current setting.")
                if path:
                    st.write("Partial path: " + " → ".join(map(str, path)))
        else:
            st.info("Click 'Run Value Iteration' to calculate the value function and optimal policy.")

    st.markdown("---")
    st.markdown("## How the algorithm works")
    st.latex(r"V(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V(s')]")
    st.write(
        "In this assignment, movement is deterministic, so each action leads to exactly one next state. "
        "The algorithm repeatedly updates each state's value until convergence, then selects the action with the highest value as the optimal policy."
    )


if __name__ == "__main__":
    main()