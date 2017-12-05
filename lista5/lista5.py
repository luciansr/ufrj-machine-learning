from __future__ import division
import numpy as np
from hmmlearn import hmm

def lista5():

    states = ["A", "B"]
    n_states = len(states)

    observations = ["H", "T"]
    n_observations = len(observations)

    start_probability = np.array([0.2, 0.8])

    transition_probability = np.array([
        [0.3, 0.7],
        [0.7, 0.3]
    ])

    emission_probability = np.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])

    model = hmm.MultinomialHMM(n_components=n_states, init_params="", n_iter=100)
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    # predict a sequence of hidden states based on visible states
    bob_says = np.array([[0,1,0,1,0,1,0,1,0,1,0]]).T

    # alice_hears = model.predict(bob_says)
    logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
    print ("Bob says:", ", ".join(map(lambda x: observations[x[0]], bob_says)))
    print ("Alice hears:", ", ".join(map(lambda x: states[x], alice_hears)))

    log_prob_obs = model.score(bob_says)
    print("log prob: " , log_prob_obs, " prob:", np.exp(log_prob_obs))
    print("probabilities")
    print(model.predict_proba(bob_says))

def lista5_6():

    states = ["Hot", "Cold"]
    n_states = len(states)

    observations = ["Small", "Medium", "Large"]
    n_observations = len(observations)

    start_probability = np.array([0.5, 0.5])

    transition_probability = np.array([
        [0.75, 0.25],
        [0.6, 0.4]
    ])

    emission_probability = np.array([
        [0.05, 0.4, 0.55],
        [0.8, 0.1, 0.1]
    ])

    model = hmm.MultinomialHMM(n_components=n_states, init_params="", n_iter=100)
    model.startprob_ = start_probability
    model.transmat_ = transition_probability
    model.emissionprob_ = emission_probability

    # predict a sequence of hidden states based on visible states
    



    sequence = 1
    for item0 in range(2):
        for item1 in range(2):
            for item2 in range(2):
                for item3 in range(2):
                    bob_says = np.array([[item0,item1,item2,item3]]).T

                    print("\nSequence: ", sequence)
                    sequence += 1

                    # alice_hears = model.predict(bob_says)
                    logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
                    print ("Bob says:", ", ".join(map(lambda x: observations[x[0]], bob_says)))
                    print ("Alice hears:", ", ".join(map(lambda x: states[x], alice_hears)))

                    log_prob_obs = model.score(bob_says)
                    print("log prob: " , log_prob_obs, " prob:", np.exp(log_prob_obs))
                    print("probabilities")
                    print(model.predict_proba(bob_says))

lista5_6()