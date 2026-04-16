from src.environment.pong_rl_environment import pong_environment
from src.agent.deepQ_agent import my_agent

STATE_SIZE = 8
ACTION_SIZE = 3

def main():
    env = pong_environment(render=True)
    agent = my_agent(STATE_SIZE, ACTION_SIZE, loadmodel=True, trainme=False, filename="pong.keras")

    state, reward, reward_left, done = env.one_step(2, human=True)

    while True:
        action = agent.get_action(state)
        next_state, reward, reward_left, done = env.one_step(action, human=True)
        state = next_state

        if done:
            state, reward, reward_left, done = env.one_step(2, human=True)

if __name__ == "__main__":
    main()