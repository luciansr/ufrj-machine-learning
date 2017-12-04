from __future__ import division
import numpy as np
from hmmlearn import hmm

def questao4_1():

    states = ["A", "B"]
    n_states = len(states)

    observations = ["H", "T"]
    n_observations = len(observations)

    start_probability = np.array([0.5, 0.5])

    transition_probability = np.array([
        [0.3, 0.7],
        [0.7, 0.3]
    ])

    emission_probability = np.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])

    model = hmm.MultinomialHMM(n_components=n_states, init_params="")
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    # predict a sequence of hidden states based on visible states
    bob_says = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]).T
    model = model.fit(bob_says)
    logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
    print ("Bob says:", ", ".join(map(lambda x: observations[x[0]], bob_says)))
    print ("Alice hears:", ", ".join(map(lambda x: states[x], alice_hears)))


questao4_1()