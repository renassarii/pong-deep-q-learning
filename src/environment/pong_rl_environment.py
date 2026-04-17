import os
import random
import pygame

pygame.init()

WIDTH, HEIGHT = 700, 500
WIN = None

FPS = 160

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_RADIUS = 7

SCORE_FONT = pygame.font.SysFont("comicsans", 50)
WINNING_SCORE = 100000


class Paddle:
    COLOR = WHITE
    VEL = 11

    def __init__(self, x, y, width, height):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.width = width
        self.height = height

    def draw(self, win):
        pygame.draw.rect(win, self.COLOR, (self.x, self.y, self.width, self.height))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y


class Ball:
    MAX_VEL = 12
    COLOR = WHITE

    def __init__(self, x, y, radius):
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.radius = radius
        self.x_vel = random.choice([-1, 1]) * self.MAX_VEL
        self.y_vel = random.uniform(-0.6 * self.MAX_VEL, 0.6 * self.MAX_VEL)

    def draw(self, win):
        pygame.draw.circle(win, self.COLOR, (int(self.x), int(self.y)), self.radius)

    def move(self):
        self.x += self.x_vel
        self.y += self.y_vel

    def reset(self):
        self.x = self.original_x
        self.y = self.original_y
        self.x_vel = random.choice([-1, 1]) * self.MAX_VEL
        self.y_vel = random.uniform(-0.6 * self.MAX_VEL, 0.6 * self.MAX_VEL)


def draw(win, paddles, ball, left_score, right_score):
    win.fill(BLACK)

    left_score_text = SCORE_FONT.render(f"{left_score}", True, WHITE)
    right_score_text = SCORE_FONT.render(f"{right_score}", True, WHITE)

    win.blit(left_score_text, (WIDTH // 4 - left_score_text.get_width() // 2, 20))
    win.blit(right_score_text, (int(WIDTH * 3 / 4) - right_score_text.get_width() // 2, 20))

    for paddle in paddles:
        paddle.draw(win)

    for i in range(10, HEIGHT, HEIGHT // 20):
        if i % 2 == 1:
            continue
        pygame.draw.rect(win, WHITE, (WIDTH // 2 - 5, i, 10, HEIGHT // 20))

    ball.draw(win)
    pygame.display.update()


def handle_collision(ball, left_paddle, right_paddle):
    reward_right = 0
    reward_left = 0

    if ball.y + ball.radius >= HEIGHT:
        ball.y_vel *= -1
    elif ball.y - ball.radius <= 0:
        ball.y_vel *= -1

    if ball.x_vel < 0:
        if left_paddle.y <= ball.y <= left_paddle.y + left_paddle.height:
            if ball.x - ball.radius <= left_paddle.x + left_paddle.width:
                ball.x_vel *= -1

                middle_y = left_paddle.y + left_paddle.height / 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (left_paddle.height / 2) / ball.MAX_VEL
                ball.y_vel = -difference_in_y / reduction_factor
                reward_left = 0.2
    else:
        if right_paddle.y <= ball.y <= right_paddle.y + right_paddle.height:
            if ball.x + ball.radius >= right_paddle.x:
                ball.x_vel *= -1

                middle_y = right_paddle.y + right_paddle.height / 2
                difference_in_y = middle_y - ball.y
                reduction_factor = (right_paddle.height / 2) / ball.MAX_VEL
                ball.y_vel = -difference_in_y / reduction_factor
                reward_right = 0.2

    return reward_right, reward_left


def handle_paddle_movement(keys, actionrightpaddle, human, actionleftpaddle, left_paddle, right_paddle):
    if human:
        if keys[pygame.K_w] and left_paddle.y - left_paddle.VEL >= 0:
            left_paddle.move(up=True)
        if keys[pygame.K_s] and left_paddle.y + left_paddle.VEL + left_paddle.height <= HEIGHT:
            left_paddle.move(up=False)
    else:
        if actionleftpaddle == 0 and left_paddle.y - left_paddle.VEL >= 0:
            left_paddle.move(up=True)
        if actionleftpaddle == 1 and left_paddle.y + left_paddle.VEL + left_paddle.height <= HEIGHT:
            left_paddle.move(up=False)

    if actionrightpaddle == 0 and right_paddle.y - right_paddle.VEL >= 0:
        right_paddle.move(up=True)
    if actionrightpaddle == 1 and right_paddle.y + right_paddle.VEL + right_paddle.height <= HEIGHT:
        right_paddle.move(up=False)


class pong_environment:
    def __init__(self, **kwargs):
        global WIN

        self.render = kwargs.get("render", False)

        if self.render:
            print("Rendering on")
            if pygame.display.get_surface() is None:
                WIN = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Pong")
            else:
                WIN = pygame.display.get_surface()
        else:
            print("Rendering off")

        self.clock = pygame.time.Clock()
        self.left_paddle = Paddle(10, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle = Paddle(WIDTH - 10 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = Ball(WIDTH // 2, HEIGHT // 2, BALL_RADIUS)

        self.left_score = 0
        self.right_score = 0

        if self.render:
            draw(WIN, [self.left_paddle, self.right_paddle], self.ball, self.left_score, self.right_score)

    def one_step(self, actionrightpaddle, human=True, actionleftpaddle=2):
        if self.render:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

            draw(WIN, [self.left_paddle, self.right_paddle], self.ball, self.left_score, self.right_score)

        if human:
            keys = pygame.key.get_pressed()
        else:
            keys = []

        handle_paddle_movement(
            keys,
            actionrightpaddle,
            human,
            actionleftpaddle,
            self.left_paddle,
            self.right_paddle
        )

        self.ball.move()

        reward_right = 0
        reward_left = 0

        hit_reward_right, hit_reward_left = handle_collision(self.ball, self.left_paddle, self.right_paddle)
        reward_right += hit_reward_right
        reward_left += hit_reward_left

        if self.ball.x < 0:
            self.right_score += 1
            reward_right += 1
            reward_left -= 1
            self.ball.reset()
        elif self.ball.x > WIDTH:
            self.left_score += 1
            reward_right -= 1
            reward_left += 1
            self.ball.reset()

        won = False
        if self.left_score >= WINNING_SCORE:
            won = True
            win_text = "Left Player Won!"
        elif self.right_score >= WINNING_SCORE:
            won = True
            win_text = "Right Player Won!"

        if won:
            if self.render:
                text = SCORE_FONT.render(win_text, True, WHITE)
                WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2))
                pygame.display.update()
                pygame.time.delay(1000)

            self.ball.reset()
            self.left_paddle.reset()
            self.right_paddle.reset()
            self.left_score = 0
            self.right_score = 0

        state = [
            self.ball.x / WIDTH,
            self.ball.y / HEIGHT,
            self.ball.x_vel / self.ball.MAX_VEL,
            self.ball.y_vel / self.ball.MAX_VEL,
            self.left_paddle.x / WIDTH,
            self.left_paddle.y / HEIGHT,
            self.right_paddle.x / WIDTH,
            self.right_paddle.y / HEIGHT,
        ]

        done = abs(reward_right) >= 1 or abs(reward_left) >= 1
        return state, reward_right, reward_left, done