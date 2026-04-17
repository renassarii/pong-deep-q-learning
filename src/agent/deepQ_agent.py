import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras


class my_agent:
    def __init__(self, inp_shape, output_shape, loadmodel=False, trainme=True, filename="models/pong.keras"):
        self.GAMMA = 0.97
        self.EPSILON = 1.0
        self.EPSILON_MIN = 0.001
        self.EPSILON_DECAY = 0.999

        self.BATCH_SIZE = 64
        self.MEMORY_SIZE = 100000
        self.LEARNING_RATE = 0.001
        self.TRAIN_START = 100

        self.inp_shape = inp_shape
        self.output_shape = output_shape
        self.model_filename = filename

        self.step = 0
        self.n_update_target_model = 1000
        self.n_save_model = 50

        if not trainme:
            self.EPSILON = 0.0
            self.EPSILON_MIN = 0.0

        self.memory = deque(maxlen=self.MEMORY_SIZE)
        self.loss_fn = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)

        self.model = self.load_or_create_model(loadmodel)
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.save_model()

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.inp_shape,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(self.output_shape, activation="linear")
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
            loss="mse"
        )
        return model

    def load_or_create_model(self, loadmodel):
        if loadmodel and os.path.isfile(self.model_filename):
            try:
                print(f"Lade Modell: {self.model_filename}", flush=True)
                model = keras.models.load_model(self.model_filename, compile=False)

                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
                    loss="mse"
                )

                print(f"Modell erfolgreich geladen: {self.model_filename}", flush=True)
                return model

            except Exception as e:
                print(f"Fehler beim Laden von {self.model_filename}: {e}", flush=True)
                print("Erstelle neues Modell...", flush=True)
        else:
            print(f"Kein Modell gefunden: {self.model_filename}", flush=True)
            print("Erstelle neues Modell...", flush=True)

        return self.build_model()

    def save_model(self):
        folder = os.path.dirname(self.model_filename)
        if folder:
            os.makedirs(folder, exist_ok=True)

        self.model.save(self.model_filename)
        print(f"Modell gespeichert: {self.model_filename}", flush=True)

    def update_target_model_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY
            self.EPSILON = max(self.EPSILON, self.EPSILON_MIN)

        if np.random.rand() <= self.EPSILON:
            return random.randrange(self.output_shape)

        state_array = np.asarray(state, dtype=np.float32).reshape(1, -1)
        q_values = self.model(state_array, training=False).numpy()[0]
        return int(np.argmax(q_values))

    def train(self):
        if len(self.memory) < max(self.TRAIN_START, self.BATCH_SIZE):
            return

        if self.step % 10 == 0:
            print(f"train step: {self.step} | epsilon: {self.EPSILON:.5f}", flush=True)

        minibatch = random.sample(self.memory, self.BATCH_SIZE)

        states = np.array([item[0] for item in minibatch], dtype=np.float32)
        actions = np.array([item[1] for item in minibatch], dtype=np.int32)
        rewards = np.array([item[2] for item in minibatch], dtype=np.float32)
        next_states = np.array([item[3] for item in minibatch], dtype=np.float32)
        dones = np.array([item[4] for item in minibatch], dtype=np.float32)

        next_q_values = self.target_model(next_states, training=False).numpy()
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1.0 - dones) * self.GAMMA * max_next_q_values

        with tf.GradientTape() as tape:
            all_q_values = self.model(states, training=True)
            chosen_q_values = tf.gather(all_q_values, actions, batch_dims=1)
            loss = tf.reduce_mean(self.loss_fn(target_q_values, chosen_q_values))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.step += 1

        if self.step % self.n_update_target_model == 0:
            self.update_target_model_weights()
            print("Target-Model aktualisiert.", flush=True)

        if self.step % self.n_save_model == 0:
            self.save_model()