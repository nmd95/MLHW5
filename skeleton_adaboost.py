#################################
# Your name: Nimrod de la Vega
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    print(4)
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n = len(X_train)

    def D1(i):
        return 1 / n

    distributions = [D1]
    alphas = []
    weak_learners = []
    for t in range(T):
        h_t = learn_weakly(X_train, y_train, distributions[-1])
        weak_learners.append(h_t)
        coordinate = h_t[0]
        projections = [v[coordinate] for v in X_train]
        epsilon_t = calc_weighted_error(projections, h_t[1], distributions[-1], y_train)
        if epsilon_t != 0:
            alpha_t = 0.5 * np.exp(np.log(((1 - epsilon_t) / epsilon_t)))
        else:
            alpha_t = 0.5
        alphas.append(alpha_t)

        def plus_minus(boolean):
            if not boolean:
                return -1
            elif boolean:
                return 1

        label_predictions = map(plus_minus, [(h_t[1])(proj) for proj in projections])
        dist_plus_one = d_t_plus_one(distributions[-1], epsilon_t, label_predictions, y_train)
        distributions.append(dist_plus_one)

    def determine_side(f, theta):
        if not f(theta - 1):
            return -1
        elif f(theta - 1):
            return 1

    hypotheses = []
    for h in weak_learners:
        theta = h[3]
        tri_tup = (determine_side(h[1], theta), h[0], theta)
        hypotheses.append(tri_tup)
    return hypotheses, alphas

##############################################
# You can add more methods here, if needed.
def calc_weighted_error(projected_samples, decision_stump, distribution, labels):
    print(3)
    truth_list = [decision_stump(projection) for projection in projected_samples]
    weighted_error = 0
    for i, v_l_tup in enumerate(zip(truth_list, labels)):
        value = v_l_tup[0]
        label = v_l_tup[1]
        if ((not value) and (label > 0)) or (value and (label < 0)):
            weighted_error += distribution(i)
    return weighted_error


def learn_weakly(vectors, their_labels, distribution):  # return a tuple : (coordinate targeted by learner,best weak
    # learner as threshold function function, it's error, theta).
    print(2)
    best_stump = (0, lambda x: x, 2, 0)  # (coordinate, decision_stump, error, theta)
    for coordinate in range((len(vectors[0]))):
        ith_projection = np.array([v[coordinate] for v in vectors])
        best_axis_stump = (lambda x: x, 2, 0)  # (decision_stump, error)
        for projection in ith_projection:
            weighted_error_theta_smaller = calc_weighted_error(ith_projection, lambda x: x < projection, distribution,
                                                               their_labels)
            weighted_error_theta_bigger = calc_weighted_error(ith_projection, lambda x: x >= projection, distribution,
                                                              their_labels)
            if best_axis_stump[1] > weighted_error_theta_bigger:
                best_axis_stump = (lambda x: x >= projection, weighted_error_theta_bigger, projection)
            if best_axis_stump[1] > weighted_error_theta_smaller:
                best_axis_stump = (lambda x: x < projection, weighted_error_theta_smaller, projection)

        if best_stump[2] > best_axis_stump[1]:
            best_stump = (coordinate, best_axis_stump[0], best_axis_stump[1], best_axis_stump[2])
    return best_stump


def d_t_plus_one(d_t, weighted_error_t, label_predictions, actual_labels):
    print(1)
    z_t = 0
    for index, p_l_tup in enumerate(zip(label_predictions, actual_labels)):
        pred = p_l_tup[0]
        labe = p_l_tup[1]
        if weighted_error_t != 0:
            alpha = 0.5 * np.log(((1 - weighted_error_t) / weighted_error_t))
        else:
            alpha = 0.5
        z_t += ((np.exp(alpha * (-1 * pred * labe))) * d_t(index))

    def d_new(index):
        return ((d_t(index)) * (
            np.exp(alpha * (-1 * pred * labe)))) / z_t

    return d_new


##############################################


def main():

    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    effective_x = X_train
    effective_y = y_train
    hypotheses, alpha_vals = run_adaboost(effective_x, effective_y, 1)

    print("x_train:")
    print(effective_x)
    print("labels :")
    print(effective_y)
    print(hypotheses)
    print(alpha_vals)
    ##############################################
    # You can add more methods here, if needed.



##############################################


if __name__ == '__main__':
    main()
