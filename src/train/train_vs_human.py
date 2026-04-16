from src.environment.pong_rl_environment import pong_environment
from src.agent.deepQ_agent import my_agent

EPISODES = 1000
STATE_SIZE = 8
ACTION_SIZE = 3

def main():
    env = pong_environment(render=True)
    agent = my_agent(STATE_SIZE, ACTION_SIZE, loadmodel=False, trainme=True, filename="pong.keras")

    for episode in range(EPISODES):
        state, reward, reward_left, done = env.one_step(2, human=True)
        total_reward = 0
        steps = 0

        while not done:
            action = agent.get_action(state)

            next_state, reward, reward_left, done = env.one_step(action, human=True)

            agent.memory.append((state, action, reward, next_state, done))
            agent.train()

            state = next_state
            total_reward += reward
            steps += 1

            if steps % 100 == 0:
                print(f"Episode {episode} | Steps {steps} | Reward {total_reward}")

        print(f"Episode {episode} finished | Total reward: {total_reward}")

if __name__ == "__main__":
    main()