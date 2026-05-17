import re

WIN = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for a, b2, c in WIN:
        if b[a] != " " and b[a] == b[b2] == b[c]:
            return b[a]
    return None


def parse(s):
    m = re.search(r"\[(\d+)\]|\b(\d+)\b", s)
    return int(m.group(1) or m.group(2)) if m else None


async def agent_loop(data, context):
    system_prompt = """You are playing Tic Tac Toe. Your goal is to win three in a row (horizontally, vertically, or diagonally) on the board.
On your turn, you should select the square number (0-8) you want to put your mark in next. For example, '[4]' places your mark in the center cell of the board.
Board:

 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8

Do not explain. Do not say anything except the action."""

    llm_client = context["client"]
    models = [data["model_a"], data["model_b"]]
    chats = [
        [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': 'Your turn.'}],
        [{'role': 'system', 'content': system_prompt}]
    ]
    rewards = [0, 0]

    board = [" "] * 9
    pid = 0

    while True:
        response = (await llm_client.responses.create(model=models[pid], input=chats[pid])).output[-1].content[0].text
        print(pid, response)
        a = parse(response)
        chats[pid].append({'role': 'assistant', 'content': response})

        if a is None or not (0 <= a <= 8) or board[a] != " ":  # invalid pid lose
            rewards[pid] = -1
            break

        board[a] = "O" if pid == 0 else "X"
        w = winner(board)

        if w:  # pid win
            rewards[pid] = 1
            rewards[1 - pid] = -1
            break
        if " " not in board:  # tie
            break

        chats[1 - pid].append({"role": "user", "content": f"Opponent played [{a}]. Your turn."})
        pid = 1 - pid

    print('Reward:', rewards)
    return [
        {'reward': rewards[0], 'chat': chats[0], 'model_name': models[0]},
        {'reward': rewards[1], 'chat': chats[1], 'model_name': models[1]},
    ]

