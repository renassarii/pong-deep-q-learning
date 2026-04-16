import numpy as np
from collections import deque
import random

import tensorflow as tf
from tensorflow import keras

class my_agent:
    def __init__(self,inp_shape, output_shape,loadmodel=False,trainme=True,filename="pong.keras"):

        self.GAMMA = 0.97  #parameter in deep-Q formula for importance of the future steps
        self.EPSILON = 1.0 #epsilon greedy strategy
        self.EPSILON_MIN = 0.001
        self.EPSILON_DECAY = 0.999
        self.BATCH_SIZE = 32 #number of random sampled items from memory that are used for a training step
        self.MEMORY_SIZE = 100000 
        self.LEARNING_RATE = 0.001 #learning rate of the gradient descent algorithm (<=0.001)
        self.TRAIN_START = 100  # don't train before memory has 100 memory items
        self.inp_shape=inp_shape #number of input neurons
        self.output_shape=output_shape #number of output neurons (=number of actions)
        self.step=0
        self.n_update_target_model=1000 # update weights of target model every 1000 steps
        self.model_filename=filename
        self.n_save_model=150

        if not trainme:
            self.EPSILON=0
            self.EPSILON_MIN=0

        if loadmodel:
            self.model = tf.keras.models.load_model(self.model_filename)
        else: 
            self.model = self.build_model(inp_shape, output_shape)

        #Using a target model stabilitzes the training process against overoscillations or slow learning
        self.target_model = self.build_model(inp_shape, output_shape)
        self.target_model.set_weights(self.model.get_weights()) #target model starts with the same weights as trained model

        
        self.memory = deque(maxlen=self.MEMORY_SIZE) #deque automatically limits the length of the memory
        self.loss_fn = keras.losses.Huber() #loss function that is faster than mean squared error
        self.optimizer = keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)  

    # --- Build Q-network ---
    def build_model(self,inp_shape, output_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(inp_shape,)))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(output_shape, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE))
        return model

    def train(self):

        if len(self.memory)>self.TRAIN_START: #only train if memory is already large enough
            print("train step:",self.step," epsilon:",self.EPSILON)
            # sample random choice of states and rewards from memory and convert them to numpy arrays
            minibatch = random.sample(self.memory, self.BATCH_SIZE)
            states, actions, rewards, next_states,done = [
                    np.array([data[i] for data in minibatch])
                    for i in range(5)
                ] #Extrahieren der einzelnen Datenkategorien aus memory und Konvertierung in numpy array
            
            #use target model to calculate future Qvales/rewards
            next_Q_values = self.target_model.predict(next_states,verbose=0)
            max_next_Q_values = np.max(next_Q_values, axis=1)
            
            #(1-done) yields 0 if done=True. If game is done, there is no useful next_states
            #but we still want to collect the reward
            target_Q_values = (rewards +(1-done)* self.GAMMA * max_next_Q_values) #Q-learning Formel
        
            
            tape=tf.GradientTape()
            with tf.GradientTape() as tape: #GradientTape ermöglicht automatisches Differenzieren und die Verfolgung der Gradienten während der Vorwärtsausführung
                allQvalues = self.model(states, training=True)
                Qvalues = tf.gather(allQvalues,actions,batch_dims=1) #only take Qvalues corresponding to the action
                loss = tf.reduce_mean(self.loss_fn(target_Q_values, Qvalues)) #Loss wird berechnet (gradienten werden nachher mitberechnet durch aktives tape)
            grads = tape.gradient(loss, self.model.trainable_variables) #gradienten für loss werden für alle Gewichte und Biase in grads gespeichert.
            self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables)) #Gewichte und Biase werden anhand der Gradienten angepasst. (nur ein Schritt Richtung Minimum pro Trainingsdurchlauf!)
            
            if self.step % self.n_update_target_model == 0:
                self.update_target_model_weights()
            
            if self.step%self.n_save_model==0:
                self.model.save(self.model_filename)
                print("saved model:"+ self.model_filename)
            
            self.step=self.step+1

    def get_action(self,state):
        #Epsilon greedy strategy
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

        if np.random.rand() <= self.EPSILON:
            return random.randrange(self.output_shape) # random move
        q_values = self.model.predict(np.asarray(state)[np.newaxis], verbose=0)[0] #AI model prediction
        return np.argmax(q_values)
    
    def update_target_model_weights(self):
        #update weights of target model with weights of trained model
        self.target_model.set_weights(self.model.get_weights())

