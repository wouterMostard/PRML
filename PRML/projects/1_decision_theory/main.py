import streamlit as st
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def create_data() -> np.ndarray:
    return np.concatenate([
        np.random.normal(mean_1, std_1, num_1).flatten(),
        np.random.normal(mean_2, std_2, num_2).flatten(),
    ]).reshape(-1, 1), np.concatenate([np.zeros(num_1).flatten(), np.ones(num_2).flatten()]).reshape(-1, 1).astype(int)

def calculate_priors(labels: np.ndarray) -> Dict[int, float]:
    labels, counts = np.unique(labels, return_counts=True)
    counts = counts /  counts.sum()

    return dict(zip(labels, counts))

def create_class_conditional_distribution(X_class):
    return stats.norm(X_class.mean(), X_class.std())


with st.sidebar:
    mean_1 = st.slider(label='mean class 1', step=0.1, value=1.0, min_value=-3.0, max_value=3.0)
    num_1 = st.slider(label='num class 1', step=50, value=1000, min_value=50, max_value=1500)
    std_1 = st.slider(label='std class 1', step=0.1, value=1.0, min_value=0.0, max_value=3.0)

    mean_2 = st.slider(label='mean class 2', step=0.1, value=1.0, min_value=-3.0, max_value=3.0)
    num_2 = st.slider(label='num class 2', step=50, value=1000, min_value=50, max_value=1500)
    std_2 = st.slider(label='std class 2', step=0.1, value=1.0, min_value=.0, max_value=3.0)

X, y = create_data()

fig, ax = plt.subplots()

priors = calculate_priors(y)
class_conditionals = [create_class_conditional_distribution(X[y == class_]) for class_ in np.unique(y)]
x_range = np.linspace(-5, 5, 300)


p_x = np.zeros(x_range.size)

for idx, class_conditional in enumerate(class_conditionals):
    ax.plot(x_range, class_conditional.pdf(x_range) * priors[idx])

    p_x += class_conditional.pdf(x_range) * priors[idx]

st.pyplot(fig)

a = -10
b = 10
N = x_range.size

I = (p_x * ((b - a) / N)).sum()

fig_p_data, ax_p_data = plt.subplots()

ax_p_data.plot(x_range, p_x / I)

st.pyplot(fig_p_data)

input_val = 0

posteriors = []

for class_k in range(len(priors)):
    posteriors.append(class_conditionals[class_k].pdf(input_val) * priors[class_k])

posteriors = np.array(posteriors)
posteriors = posteriors / posteriors.sum()

