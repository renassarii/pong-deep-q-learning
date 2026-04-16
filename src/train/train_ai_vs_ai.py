from src.environment.pong_rl_environment import pong_environment
from src.agent.deepQ_agent import my_agent

EPISODES = 1000
STATE_SIZE = 8
ACTION_SIZE = 3

def main():
    env = pong_environment(render=True)

    agent_right = my_agent(STATE_SIZE, ACTION_SIZE, loadmodel=False, trainme=True, filename="pong_right.keras")
    agent_left = my_agent(STATE_SIZE, ACTION_SIZE, loadmodel=False, trainme=True, filename="pong_left.keras")

    for episode in range(EPISODES):
        state, reward_right, reward_left, done = env.one_step(2, human=False, actionleftpaddle=2)

        total_reward_right = 0
        total_reward_left = 0
        steps = 0

        while not done:
            action_right = agent_right.get_action(state)
            action_left = agent_left.get_action(state)

            next_state, reward_right, reward_left, done = env.one_step(
                action_right,
                human=False,
                actionleftpaddle=action_left
            )

            agent_right.memory.append((state, action_right, reward_right, next_state, done))
            agent_left.memory.append((state, action_left, reward_left, next_state, done))

            agent_right.train()
            agent_left.train()

            state = next_state
            total_reward_right += reward_right
            total_reward_left += reward_left
            steps += 1

            if steps % 100 == 0:
                print(
                    f"Episode {episode} | Steps {steps} | "
                    f"Right reward {total_reward_right} | Left reward {total_reward_left}"
                )

        print(
            f"Episode {episode} finished | "
            f"Right total reward: {total_reward_right} | Left total reward: {total_reward_left}"
        )

if __name__ == "__main__":
    main()