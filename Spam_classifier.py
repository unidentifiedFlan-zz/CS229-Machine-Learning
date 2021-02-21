import numpy as np
import pandas as pd

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = []
    N = matrix.shape[0]
    V = matrix.shape[1]
    ###################

    #In naive bayes we assume each word feature is independent given the classification, y, of the email.
    #In training, p(xi=1|y=1) = sum(xi=1 && y =1)/sum(y=1)
    #p(xi=1|y=0) = sum(xi=1 && y=0)/sum(y=0)
    #p(y=1) = sum(y=1)/n, n=number of data samples
    #
    #To classify new data word vectors x, p(y=1|x)=p(x|y=1)*p(y=1)/p(x)
    # =>  p(y=1|x) = PROD(p(xi|y=1))*p(y=1)/( PROD(p(xi|y=1))p(y=1) + PROD(p(xi|y=0))*p(y=0))
    #
    #The above is for a binomial distribution where xj=1 or 0. We require a multinomial distribution
    #where xj=0,1,2,3,...
    # For this, with Laplace smoothing, we can use the formulae at the end of notes2
    #
    # p(k|y=1) = (1 + sum[n](sum[d]( 1(xj = k) )))/(V + sum[n]( 1(y=1) ))
    #where k is the kth word in the dictionary, n is number of samples, d is number of words in sample
    #xj is jth word in sample, V is size of dictionary
    #

    #The input matrix consists of n rows and V columns. Each entry is the number of times word j in the dictionary
    #appears in sample i
    # Therefore we already have sum[d]( 1(xj=k) )
    #we therefore just need to sum over each column
    # For the denominator we could sum over category to get number of trues, and then n minus this to get number
    # of falses

    num_spam_emails = sum(category)
    num_non_spam_emails = N - num_spam_emails

    mod_matrix = np.c_[matrix, category]
    df = pd.DataFrame(mod_matrix)
    total_dict_freq_spam = df.sum(axis=0).where(df[V]==1.0, 0).to_numpy()
    total_dict_freq_spam = np.delete(total_dict_freq_spam, -1)

    total_dict_freq_non_spam = df.sum(axis=0).where(df[V]==0.0, 0).to_numpy()
    total_dict_freq_non_spam = np.delete(total_dict_freq_non_spam, -1)

    word_probabilites_spam = (np.ones(V) + total_dict_freq_spam)/(V + num_spam_emails)
    word_probabilites_non_spam = (np.ones(V) + total_dict_freq_non_spam)/(V + num_non_spam_emails)
    prob_spam = num_spam_emails/N
    prob_non_spam = num_non_spam_emails/N

    state.append(prob_spam)
    state.append(prob_non_spam)
    state.append(word_probabilites_spam)
    state.append(word_probabilites_non_spam)

    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################

    #p(y=1) =  p(x|y=1)*p(y=1)/(p(x|y=1)*p(y=1) + p(x|y=0)*p(y=0))
    # where p(x|y=1) = PROD(p(xj|y=1); j < d)*p(y=1)
    #
    #log(p(y=1)) = log(p(x|y=1)) + log(p(y=1)) - log[p(x|y=1) + p(y=1) + p(x|y=0) + p(y=0)]

    #cum_word_spam_probabilities = np.prod(np.power(state[2], matrix), axis=1)
    #cum_word_non_spam_probabilities = np.prod(np.power(state[3], matrix), axis=1)
    #spam_probability = cum_word_spam_probabilities*state[0]/(cum_word_spam_probabilities*state[0] + cum_word_non_spam_probabilities*state[1])
    #output = spam_probability > 0.5

    #Using logarithms
    cum_word_spam_prob_log = np.sum( np.log(state[2])*matrix, axis = 1 )
    cum_word_non_spam_prob_log = np.sum( np.log(state[3])*matrix, axis=1)

    spam_prob_log = -np.log(1 + np.exp(cum_word_non_spam_prob_log + np.log(state[1]) - cum_word_spam_prob_log - np.log(state[0])))

    output = spam_prob_log > np.log(0.5)

    output = output.astype(float)

    ###################
    return output

def evaluate(output, label):
    #error = (output != label).sum() * 1. / len(output)
    error = sum(1 for a, b in zip(output, label) if a != b) * 1. / len(output)
    print('Error: %1.4f' % error)
    return error

def main():

    training_sets = [50, 100, 200, 400, 800, 1400]
    data_path = "C:/Users/scday/Documents/coding/Machine_Learning/CS229/ps2/spam_data/spam_data/"

    matrix_train_paths = []
    for set in training_sets:
        matrix_train_paths.append([data_path + "MATRIX.TRAIN." + str(set), set])
    matrix_test_path = data_path + "MATRIX.TEST"

    errors = []
    for path, size in matrix_train_paths:
        trainMatrix, tokenlist, trainCategory = readMatrix(path)
        testMatrix, tokenlist, testCategory = readMatrix(matrix_test_path)

        state = nb_train(trainMatrix, trainCategory)
        output = nb_test(testMatrix, state)

#    indications = np.log(state[2]/state[3])
#    largest_indications_indices = indications.argsort()[-5:]
#    most_indicative_tokens = []
#    for i in range(5):
#     most_indicative_tokens.append(tokenlist[largest_indications_indices[i]])
#    print(most_indicative_tokens) #Output: ['re', 'spam', 'email', 'emailaddr', 'number']

        error = evaluate(output, testCategory)
        errors.append([error, size]) # 200 size training set gives greatest error, while 400 size gives least.

    print(errors)
    return

if __name__ == '__main__':
    main()
