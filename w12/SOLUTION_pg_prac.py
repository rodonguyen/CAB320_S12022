'''

The aim of this Prac is to test the policy gradient algorithm
REINFORCE on a the pole balancing task.

The sequence of steps between the moment the environment 
is reset until it is done is called an "episode". 
At the end of an episode 
(i.e., when step() returns done=True), 
you should reset the environment before you continue to use it.

'''


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
import gym

def plot_environment(env, figsize=(5,4)):
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    return img
    
    
# --------------  Exercise 1 --------------   

def basic_policy(obs):
    '''
    Take an observation as input, return output the action to be executed 
    following the simple strategy that accelerates left when the pole is 
    leaning toward the left and accelerates right when the pole is 
    leaning toward the right.
    
    Actions are encoded as integers:
        accelerating left (0) or right (1). 
    
    '''
    angle = obs[2]
    return 0 if angle < 0 else 1

def collect_stats_simple_strategy():
    '''
    Run 500 episodes and collect stats
    on the performance of the simple strategy 
    '''
    env = gym.make("CartPole-v1")
    env.seed(42)
        
    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        for step in range(200):
            action = basic_policy(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        totals.append(episode_rewards)    

    print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))    
    env.close()
# ---------------------------------------------   

# Visualize one episode basic strategy

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

def view_one_episode_basic():
    env = gym.make("CartPole-v1")

    env.seed(42)    
    frames = []    
    obs = env.reset()
    for step in range(200):
        img = env.render(mode="rgb_array")
        frames.append(img)
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break    
    env.close()

    print('Episode length = ', len(frames))
    plot_animation(frames)
    
# --------------  Exercise 2 --------------   

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

n_inputs = 4 # == env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])
    

    
def render_policy_net(model, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    return frames
    
def view_one_episode_initial_policy_net():
    frames = render_policy_net(model)
    print('Episode length = ', len(frames))
    plot_animation(frames)

# --------------  Exercise 3 --------------   

# Policy Gradients
    

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads


def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]
    
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_rate = 0.95
optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.binary_crossentropy

def do_training():
    keras.backend.clear_session()
    
    env = gym.make("CartPole-v1")
    env.seed(42);
    
    for iteration in range(n_iterations):
        all_rewards, all_grads = play_multiple_episodes(
            env, n_episodes_per_update, n_max_steps, model, loss_fn)
        total_rewards = sum(map(sum, all_rewards))                     # Not shown in the book
        print("\rIteration: {}, mean rewards: {:.1f}".format(          # Not shown
            iteration, total_rewards / n_episodes_per_update), end="") # Not shown
        all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                           discount_rate)
        all_mean_grads = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [final_reward * all_grads[episode_index][step][var_index]
                 for episode_index, final_rewards in enumerate(all_final_rewards)
                     for step, final_reward in enumerate(final_rewards)], axis=0)
            all_mean_grads.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
    
    env.close()    
    

# ------------------------------------------   

if __name__=='__main__':
    collect_stats_simple_strategy()    
#    view_one_episode_basic()
#    view_one_episode_initial_policy_net()
#    do_training()
