import pygame
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
import numpy as np
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")
background_image = pygame.image.load(r"./background.png")  # Load the background image
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))  # Scale to screen size

background_image2 = pygame.image.load(r"./5628127.png")  # Load the background image
background_image2 = pygame.transform.scale(background_image2, (WIDTH, HEIGHT))  # Scale to screen size

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Font for displaying score
font = pygame.font.Font(None, 36)

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Bird settings
bird = pygame.Rect(50, 300, 30, 30)
bird_velocity = 0
gravity = 0.5

# Pipe settings
pipe_width = 50
pipe_gap = 150
pipe_velocity = -3

# Load assets
bird_image = pygame.image.load(
    r"C://Users//dell//Downloads//birdimage.png"
)
bird_image = pygame.transform.scale(bird_image, (30, 30))  # Resize bird to 30x30 pixels
pipe_image = pygame.image.load(r"./images.jpg")

jump_sound = pygame.mixer.Sound(
    r"./flap-101soundboards.mp3"
)
collision_sound = pygame.mixer.Sound(
    r"./flappy-bird-hit-sound-101soundboards.mp3"
)

# Create a neural network model
model = Sequential(
    [
        Dense(128, input_shape=(4,)),  # Increased number of neurons
        LeakyReLU(alpha=0.01),  # Leaky ReLU activation function
        Dropout(0.3),  # Increased dropout rate
        Dense(64),  # Added another dense layer with fewer neurons
        LeakyReLU(alpha=0.01),
        Dropout(0.3),
        Dense(32),  # Further reduced neurons to create a funnel effect
        LeakyReLU(alpha=0.01),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),  # Output layer with sigmoid activation
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy")


# Scoring variables
score = 0
high_score = 0


def get_state(bird, pipes, bird_velocity):
    bird_y = bird.y / HEIGHT  # Normalize
    bird_velocity /= 10  # Normalize
    pipe_x = pipes[0][0].x / WIDTH  # Normalize
    pipe_gap_y = pipes[0][0].height / HEIGHT  # Normalize
    return np.array([bird_y, bird_velocity, pipe_x, pipe_gap_y])


def get_reward(bird, pipes):
    if (
        bird.colliderect(pipes[0][0])
        or bird.colliderect(pipes[0][1])
        or bird.y > HEIGHT
    ):
        return -1  # Collision penalty
    return 0.1  # Reward for staying alive


def create_pipe():
    height = random.randint(100, 400)
    top_pipe = pygame.Rect(WIDTH, 0, pipe_width, height)
    bottom_pipe = pygame.Rect(
        WIDTH, height + pipe_gap, pipe_width, HEIGHT - height - pipe_gap
    )
    return top_pipe, bottom_pipe


def draw_bird_and_pipes(bird, pipes):
    screen.fill(WHITE)
    screen.blit(bird_image, (bird.x, bird.y))
    screen.blit(background_image, (0, 0))  # Draw the background image
    screen.blit(bird_image, (bird.x, bird.y))
    for pipe in pipes:
        top_pipe_image = pygame.transform.scale(
            pipe_image, (pipe_width, pipe[0].height)
        )
        bottom_pipe_image = pygame.transform.scale(
            pipe_image, (pipe_width, HEIGHT - pipe[0].height - pipe_gap)
        )
        screen.blit(top_pipe_image, (pipe[0].x, pipe[0].y))
        screen.blit(bottom_pipe_image, (pipe[1].x, pipe[1].y))

    # Display the score and high score
    score_text = font.render(f"Score: {score}", True, BLACK)
    high_score_text = font.render(f"High Score: {high_score}", True, BLACK)
    screen.blit(score_text, (10, 10))
    screen.blit(high_score_text, (10, 40))

    pygame.display.flip()


def reset_game():
    global bird, bird_velocity, pipes, score
    bird = pygame.Rect(50, 300, 30, 30)
    bird_velocity = 0
    pipes = [create_pipe()]
    score = 0


def collect_training_data():
    global bird_velocity, score ,high_score
    training_data = []
    reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird_velocity = -8
                if hasattr(pygame, "mixer"):
                    jump_sound.play()

        # Bird movement
        bird_velocity += gravity
        bird.y += bird_velocity

        # Pipe movement
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity

        # Remove pipes off the screen and increase score
        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())
            score += 1

        # Collision detection and reward calculation
        reward = get_reward(bird, pipes)
        state = get_state(bird, pipes, bird_velocity)
        training_data.append((state, reward))

        if reward == -1:
            if hasattr(pygame, "mixer"):
                collision_sound.play()
            running = False

        # Draw the bird and pipes
        draw_bird_and_pipes(bird, pipes)
        clock.tick(30)
    if not high_score>score:
        high_score=score;

    return training_data


def train_model(model, training_data, epochs=10):
    states, rewards = zip(*training_data)
    states = np.array(states)
    rewards = np.array(rewards)
    model.fit(states, rewards, epochs=epochs)


def automatic_play():
    global bird_velocity, score, high_score
    reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        state = get_state(bird, pipes, bird_velocity)
        action = model.predict(state.reshape(1, 4), verbose=0)[0][0]

        if action > 0.5:
            bird_velocity = -8
            if hasattr(pygame, "mixer"):
                jump_sound.play()

        bird_velocity += gravity
        bird.y += bird_velocity

        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity

        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())
            score += 1

        if score > high_score:
            high_score = score

        if (
            bird.colliderect(pipes[0][0])
            or bird.colliderect(pipes[0][1])
            or bird.y > HEIGHT
        ):
            if hasattr(pygame, "mixer"):
                collision_sound.play()
            running = False

        draw_bird_and_pipes(bird, pipes)
        clock.tick(30)


def show_menu():
    menu_running = True

    # screen.blit(bird_image, (bird.x, bird.y))
    while menu_running:
        # screen.fill(WHITE)
        screen.blit(background_image2, (0, 0))  # Draw the background image
        title_text = font.render("Flappy Bird", True, BLACK)
        play_text = font.render("Press Enter to Play", True, BLACK)
        exit_text = font.render("Press Esc to Exit", True, BLACK)

        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))
        screen.blit(play_text, (WIDTH // 2 - play_text.get_width() // 2, HEIGHT // 2))
        screen.blit(
            exit_text, (WIDTH // 2 - exit_text.get_width() // 2, HEIGHT // 2 + 50)
        )

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    menu_running = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()


def show_menu2():
    menu_running = True

    # screen.blit(bird_image, (bird.x, bird.y))
    while menu_running:
        # screen.fill(WHITE)
        # screen.blit(background_image2, (0, 0))  # Draw the background image
        title_text = font.render("Flappy Bird", True, BLACK)
        play_text = font.render("Press Enter to Play", True, BLACK)
        exit_text = font.render("Press Esc to Exit", True, BLACK)

        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))
        screen.blit(play_text, (WIDTH // 2 - play_text.get_width() // 2, HEIGHT // 2))
        screen.blit(
            exit_text, (WIDTH // 2 - exit_text.get_width() // 2, HEIGHT // 2 + 50)
        )

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    menu_running = False
                    return True
                elif event.key == pygame.K_ESCAPE:

                    return False


def main():
    show_menu()

    all_training_data = []
    print(all_training_data)
    i = True
    while i != False:
        all_training_data.extend(collect_training_data())
        i = show_menu2()

    print("Training the model")
    train_model(model, all_training_data, epochs=100)

    print("Starting automatic play")
    automatic_play()


if __name__ == "__main__":
    main()
