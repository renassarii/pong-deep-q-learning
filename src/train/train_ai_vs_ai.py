import os

RENDER = True
STATE_SIZE = 8
ACTION_SIZE = 3

RIGHT_MODEL = "models/pong_right.keras"
LEFT_MODEL = "models/pong_left.keras"

if not RENDER:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import numpy as np

from src.environment.pong_rl_environment import pong_environment
from src.agent.deepQ_agent import my_agent


def mirror_state(state):
    s = np.asarray(state, dtype=np.float32).copy()

    # Annahme:
    # [ball_x, ball_y, ball_vel_x, ball_vel_y, left_paddle_y, right_paddle_y, ...]
    s[0] = 1.0 - s[0]   # ball_x spiegeln
    s[2] = -s[2]        # ball_vel_x invertieren
    s[4], s[5] = s[5], s[4]  # paddles tauschen

    return s


def save_both_models(agent_right, agent_left):
    agent_right.save_model()
    agent_left.save_model()


def main():
    os.makedirs("models", exist_ok=True)

    env = pong_environment(render=RENDER)

    agent_right = my_agent(
        STATE_SIZE,
        ACTION_SIZE,
        loadmodel=False,
        trainme=True,
        filename=RIGHT_MODEL
    )

    agent_left = my_agent(
        STATE_SIZE,
        ACTION_SIZE,
        loadmodel=False,
        trainme=True,
        filename=LEFT_MODEL
    )

    print(f"Render mode: {RENDER}", flush=True)
    print(f"Right model path: {RIGHT_MODEL}", flush=True)
    print(f"Left model path: {LEFT_MODEL}", flush=True)

    try:
        episode = 0

        while True:
            episode += 1

            state, reward_right, reward_left, done = env.one_step(
                2,
                human=False,
                actionleftpaddle=2
            )

            total_reward_right = 0.0
            total_reward_left = 0.0
            steps = 0

            while not done:
                state_right = np.asarray(state, dtype=np.float32)
                state_left = mirror_state(state_right)

                action_right = agent_right.get_action(state_right)
                action_left = agent_left.get_action(state_left)

                next_state, reward_right, reward_left, done = env.one_step(
                    action_right,
                    human=False,
                    actionleftpaddle=action_left
                )

                next_state_right = np.asarray(next_state, dtype=np.float32)
                next_state_left = mirror_state(next_state_right)

                agent_right.memory.append(
                    (state_right, action_right, reward_right, next_state_right, done)
                )
                agent_left.memory.append(
                    (state_left, action_left, reward_left, next_state_left, done)
                )

                if steps % 4 == 0:
                    agent_right.train()
                    agent_left.train()

                state = next_state
                total_reward_right += reward_right
                total_reward_left += reward_left
                steps += 1

            print(
                f"Episode {episode} | "
                f"Steps: {steps} | "
                f"Right reward: {total_reward_right:.2f} | "
                f"Left reward: {total_reward_left:.2f} | "
                f"Right epsilon: {agent_right.EPSILON:.4f} | "
                f"Left epsilon: {agent_left.EPSILON:.4f}",
                flush=True
            )

    except KeyboardInterrupt:
        print("\nTraining abgebrochen. Speichere Modelle...", flush=True)
        save_both_models(agent_right, agent_left)
        print("Modelle gespeichert.", flush=True)


if __name__ == "__main__":
    main()