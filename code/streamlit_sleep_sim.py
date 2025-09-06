import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

# Import RL libraries
from stable_baselines3 import PPO, SAC, TD3, DDPG
import gymnasium as gym
from gymnasium import spaces

# Set page config
st.set_page_config(
    page_title="Sleep Optimization Simulator", 
    layout="wide"
)

# Sleep Environment Class
class SleepOptimizationEnv(gym.Env):
    def __init__(self, reward_type='additive', max_steps=50, safety_strict=True):
        super().__init__()
        
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.safety_strict = safety_strict
        self.current_step = 0
        
        # Action space: [delta_duration, delta_bedtime, delta_waketime]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -0.5]),
            high=np.array([0.5, 0.5, 0.5]),
            dtype=np.float32
        )
        
        # State space: [duration, cos_bed, sin_bed, cos_wake, sin_wake]
        self.observation_space = spaces.Box(
            low=np.array([4.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([12.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
    
    def _time_to_circadian(self, time_hours):
        angle = 2 * np.pi * (time_hours / 24.0)
        return np.cos(angle), np.sin(angle)
    
    def _compute_fat_loss_reward(self, duration, bedtime, waketime):
        optimal_duration = 7.5
        duration_reward = 5.0 * np.exp(-0.5 * ((duration - optimal_duration) / 1.5) ** 2)
        
        bedtime_reward = 0
        if 21.0 <= bedtime <= 23.5:
            bedtime_reward = 2.0
        elif 20.0 <= bedtime < 21.0 or 23.5 < bedtime <= 24.5:
            bedtime_reward = 1.0
            
        waketime_reward = 0
        if 6.0 <= waketime <= 8.0:
            waketime_reward = 2.0
        elif 5.5 <= waketime < 6.0 or 8.0 < waketime <= 8.5:
            waketime_reward = 1.0
        
        return duration_reward + bedtime_reward + waketime_reward
    
    def _compute_sleep_health_reward(self, duration, bedtime, waketime):
        health_reward = 1.0  # Consistency bonus
        
        if 6.5 <= duration <= 9.0:
            health_reward += 3.0
        elif 6.0 <= duration <= 10.0:
            health_reward += 1.5
        
        if 7.0 <= duration <= 8.5:
            health_reward += 1.0
        
        return health_reward
    
    def _compute_safety_penalty(self, duration, bedtime, waketime):
        penalty = 0
        
        if duration < 4.0 or duration > 12.0:
            penalty += 20.0
        elif duration < 5.0 or duration > 10.0:
            penalty += 5.0
        elif duration < 6.0 or duration > 9.5:
            penalty += 1.0
            
        if 4.0 <= bedtime <= 18.0:
            penalty += 10.0
            
        if bedtime > 2.0 and bedtime < 4.0:
            penalty += 3.0
            
        return penalty
    
    def _calculate_reward(self, duration, bedtime, waketime):
        R_fat = self._compute_fat_loss_reward(duration, bedtime, waketime)
        R_sleep = self._compute_sleep_health_reward(duration, bedtime, waketime)
        R_penalty = self._compute_safety_penalty(duration, bedtime, waketime)
        
        if self.reward_type == 'additive':
            reward = R_fat + R_sleep - R_penalty
        elif self.reward_type == 'multiplicative':
            R_fat_pos = max(0.1, R_fat)
            R_sleep_pos = max(0.1, R_sleep)
            penalty_factor = max(0.1, 1 - R_penalty/20.0)
            reward = R_fat_pos * (1 + R_sleep_pos/10.0) * penalty_factor
        
        return reward, R_fat, R_sleep, R_penalty
    
    def get_state_from_params(self, duration, bedtime, waketime):
        """Convert sleep parameters to environment state"""
        cos_bed, sin_bed = self._time_to_circadian(bedtime)
        cos_wake, sin_wake = self._time_to_circadian(waketime)
        return np.array([duration, cos_bed, sin_bed, cos_wake, sin_wake], dtype=np.float32)

# Cache models to avoid reloading
@st.cache_resource
def load_saved_models():
    """Load all saved models"""
    models = {}
    model_dir = "models"
    
    if not os.path.exists(model_dir):
        return {}
    
    algorithms = ['PPO', 'SAC', 'TD3', 'DDPG']
    reward_types = ['additive', 'multiplicative']
    algorithm_classes = {'PPO': PPO, 'SAC': SAC, 'TD3': TD3, 'DDPG': DDPG}
    
    for reward_type in reward_types:
        models[reward_type] = {}
        reward_dir = os.path.join(model_dir, reward_type)
        
        if not os.path.exists(reward_dir):
            continue
            
        for algorithm in algorithms:
            # Find the most recent model for this algorithm
            model_files = [f for f in os.listdir(reward_dir) if f.startswith(f"{algorithm}_optimized_") and f.endswith(".zip")]
            
            if model_files:
                # Sort by timestamp and get the most recent
                model_files.sort(reverse=True)
                model_path = os.path.join(reward_dir, model_files[0])
                
                try:
                    # Create environment for this reward type
                    env = SleepOptimizationEnv(reward_type=reward_type)
                    
                    # Load model
                    model = algorithm_classes[algorithm].load(model_path, env=env)
                    models[reward_type][algorithm] = {
                        'model': model,
                        'path': model_path,
                        'timestamp': model_files[0].split('_')[-1].replace('.zip', '')
                    }
                    
                except Exception as e:
                    st.error(f"Failed to load {algorithm} model for {reward_type}: {str(e)}")
    
    return models

def format_time(hours):
    """Convert decimal hours to HH:MM format"""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h:02d}:{m:02d}"

def parse_time_input(time_str):
    """Convert HH:MM format to decimal hours"""
    try:
        h, m = map(int, time_str.split(':'))
        return h + m/60.0
    except:
        return 22.0  # Default bedtime

def run_optimization_step(model, env, current_duration, current_bedtime, current_waketime):
    """Run one optimization step"""
    state = env.get_state_from_params(current_duration, current_bedtime, current_waketime)
    action, _ = model.predict(state, deterministic=True)
    
    # Apply action (small adjustments)
    new_duration = np.clip(current_duration + action[0] * 0.1, 4.0, 12.0)
    new_bedtime = (current_bedtime + action[1] * 0.1) % 24
    new_waketime = (new_bedtime + new_duration) % 24
    
    # Calculate reward for this state
    reward, R_fat, R_sleep, R_penalty = env._calculate_reward(new_duration, new_bedtime, new_waketime)
    
    return new_duration, new_bedtime, new_waketime, reward, R_fat, R_sleep, R_penalty

def plot_trajectory(trajectory):
    """Plot the optimization trajectory"""
    if len(trajectory) < 2:
        return None
    
    steps = [t['step'] for t in trajectory]
    durations = [t['duration'] for t in trajectory]
    bedtimes = [t['bedtime'] for t in trajectory]
    waketimes = [t['waketime'] for t in trajectory]
    rewards = [t['reward'] for t in trajectory]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Sleep Duration (hours)', 'Bedtime (24h)', 'Wake Time (24h)', 'Reward Score']
    )
    
    fig.add_trace(go.Scatter(x=steps, y=durations, mode='lines+markers', name='Duration'), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=bedtimes, mode='lines+markers', name='Bedtime'), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=waketimes, mode='lines+markers', name='Wake Time'), row=2, col=1)
    fig.add_trace(go.Scatter(x=steps, y=rewards, mode='lines+markers', name='Reward'), row=2, col=2)
    
    fig.update_layout(height=500, showlegend=False)
    return fig

def main():
    st.title("Sleep Optimization Simulator")
    
    # Load models
    models = load_saved_models()
    
    if not models:
        st.error("No trained models found. Please run the training script first.")
        return
    
    # Initialize session state
    if 'trajectory' not in st.session_state:
        st.session_state.trajectory = []
    if 'current_duration' not in st.session_state:
        st.session_state.current_duration = 7.5
    if 'current_bedtime' not in st.session_state:
        st.session_state.current_bedtime = 22.5
    if 'current_waketime' not in st.session_state:
        st.session_state.current_waketime = 6.0
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        available_algorithms = []
        available_reward_types = []
        
        for reward_type in models:
            if reward_type not in available_reward_types:
                available_reward_types.append(reward_type)
            for algorithm in models[reward_type]:
                if algorithm not in available_algorithms:
                    available_algorithms.append(algorithm)
        
        if not available_algorithms or not available_reward_types:
            st.error("No models available")
            return
        
        selected_algorithm = st.selectbox("Algorithm", available_algorithms)
        selected_reward_type = st.selectbox("Reward Type", available_reward_types)
        
        # Check if selected combination exists
        if selected_reward_type not in models or selected_algorithm not in models[selected_reward_type]:
            st.error(f"Model {selected_algorithm} with {selected_reward_type} reward not found")
            return
        
        st.divider()
        
        # Initial sleep parameters
        st.header("Initial Sleep Pattern")
        
        initial_duration = st.slider("Sleep Duration (hours)", 4.0, 12.0, 7.5, 0.1)
        bedtime_str = st.text_input("Bedtime (HH:MM)", "22:30")
        initial_bedtime = parse_time_input(bedtime_str)
        
        # Reset button
        if st.button("Reset to Initial Values"):
            st.session_state.current_duration = initial_duration
            st.session_state.current_bedtime = initial_bedtime
            st.session_state.current_waketime = (initial_bedtime + initial_duration) % 24
            st.session_state.trajectory = [{
                'step': 0,
                'duration': st.session_state.current_duration,
                'bedtime': st.session_state.current_bedtime,
                'waketime': st.session_state.current_waketime,
                'reward': 0,
                'R_fat': 0,
                'R_sleep': 0,
                'R_penalty': 0
            }]
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Current Sleep State")
        
        # Display current values
        current_waketime = (st.session_state.current_bedtime + st.session_state.current_duration) % 24
        st.session_state.current_waketime = current_waketime
        
        st.write(f"**Duration:** {st.session_state.current_duration:.1f} hours")
        st.write(f"**Bedtime:** {format_time(st.session_state.current_bedtime)}")
        st.write(f"**Wake Time:** {format_time(current_waketime)}")
        
        # Calculate current reward
        env = SleepOptimizationEnv(reward_type=selected_reward_type)
        reward, R_fat, R_sleep, R_penalty = env._calculate_reward(
            st.session_state.current_duration, 
            st.session_state.current_bedtime, 
            current_waketime
        )
        
        st.write(f"**Current Reward:** {reward:.2f}")
        st.write(f"- Fat Loss Component: {R_fat:.2f}")
        st.write(f"- Sleep Health Component: {R_sleep:.2f}")
        st.write(f"- Safety Penalty: {R_penalty:.2f}")
        
        # Optimization button
        if st.button("Run Optimization Step", type="primary"):
            model_info = models[selected_reward_type][selected_algorithm]
            model = model_info['model']
            
            new_duration, new_bedtime, new_waketime, new_reward, new_R_fat, new_R_sleep, new_R_penalty = run_optimization_step(
                model, env, st.session_state.current_duration, st.session_state.current_bedtime, current_waketime
            )
            
            # Update session state
            st.session_state.current_duration = new_duration
            st.session_state.current_bedtime = new_bedtime
            st.session_state.current_waketime = new_waketime
            
            # Add to trajectory
            step_num = len(st.session_state.trajectory)
            st.session_state.trajectory.append({
                'step': step_num,
                'duration': new_duration,
                'bedtime': new_bedtime,
                'waketime': new_waketime,
                'reward': new_reward,
                'R_fat': new_R_fat,
                'R_sleep': new_R_sleep,
                'R_penalty': new_R_penalty
            })
            
            st.rerun()
        
        # Clear trajectory button
        if st.button("Clear History"):
            st.session_state.trajectory = []
            st.rerun()
    
    with col2:
        st.subheader("Optimization Progress")
        
        if len(st.session_state.trajectory) > 0:
            # Show trajectory plot
            fig = plot_trajectory(st.session_state.trajectory)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show trajectory table
            if len(st.session_state.trajectory) > 1:
                st.subheader("Step History")
                
                df_data = []
                for t in st.session_state.trajectory:
                    df_data.append({
                        'Step': t['step'],
                        'Duration': f"{t['duration']:.1f}h",
                        'Bedtime': format_time(t['bedtime']),
                        'Wake Time': format_time(t['waketime']),
                        'Reward': f"{t['reward']:.2f}"
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # Show improvement
                if len(st.session_state.trajectory) > 1:
                    initial_reward = st.session_state.trajectory[0]['reward'] if st.session_state.trajectory[0]['reward'] != 0 else st.session_state.trajectory[1]['reward']
                    final_reward = st.session_state.trajectory[-1]['reward']
                    improvement = final_reward - initial_reward
                    st.write(f"**Total Improvement:** {improvement:+.2f}")
        else:
            st.info("Click 'Run Optimization Step' to start optimization")
    
    # Model info
    with st.expander("Model Information"):
        if selected_reward_type in models and selected_algorithm in models[selected_reward_type]:
            model_info = models[selected_reward_type][selected_algorithm]
            st.write(f"**Algorithm:** {selected_algorithm}")
            st.write(f"**Reward Type:** {selected_reward_type}")
            st.write(f"**Model Path:** {model_info['path']}")
            st.write(f"**Timestamp:** {model_info['timestamp']}")

if __name__ == "__main__":
    main()