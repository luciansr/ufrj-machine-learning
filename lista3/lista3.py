from sklearn.preprocessing import PolynomialFeatures
import random
import tensorflow as tf
import numpy as np


def getListFromFile(fileName):
    file_object = open(fileName, 'r')

    lista = file_object.readlines()
    result = list(lista)

    file_object.close()

    return result


def randomizeFile(inputFile, output, check):
    lista = getListFromFile(inputFile)

    random.shuffle(lista)

    outputReader = open(output, 'w')
    checkReader = open(check, 'w')

    size = len(lista)
    index = 0
    for item in lista:
        index += 1
        if(index > size - 172):
            checkReader.write(item)
        else:
            outputReader.write(item)

    outputReader.close()
    checkReader.close()


def getData():
    dataFile = 'lista3/data.txt'
    data = getListFromFile(dataFile)
    data = list(map(lambda x: list(map(float, x.split())), data))
    return {
        "data": data,
        "idade": list(map(lambda x: x[0], data)),
        "peso": list(map(lambda x: x[1], data)),
        "carga": list(map(lambda x: x[2], data)),
        "vo2": list(map(lambda x: x[3], data))
    }


def formulaGradient():
    fileData = getData()

    matriz = fileData['data']
    idade = fileData['idade']
    peso = fileData['peso']
    carga = fileData['carga']
    vo2 = fileData['vo2']

    W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(260.0)
    # (carga ×11.4+260+peso×3, 5)/peso
    hypothesis = (W1 * carga + W2 * peso + b)/peso

    cost = tf.reduce_mean(tf.square(hypothesis - vo2))

    a = tf.Variable(0.1)
    optimizer = tf.train.GradientDescentOptimizer(a)
    train = optimizer.minimize(cost)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for step in range(1,20001):
        sess.run(train)
        if step % 200 == 0:
            print (step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b))


def questao1():
    fileData = getData()

    matriz = fileData['data']
    idade = fileData['idade']
    peso = fileData['peso']
    carga = fileData['carga']
    vo2 = fileData['vo2']

    X = [peso, carga]

    poly = PolynomialFeatures(degree=2)

    X_transformed = poly.fit_transform(X)

    model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
    # y = 3 - 2 * x + x ** 2 - x ** 3
    y = (W1 * carga + W2 * peso + 260)/peso # hypothesis
    model = model.fit(x[:, np.newaxis], y)

    model.named_steps['linear'].coef_

    print(X_transformed)

questao1()


# randomizeFile('lista3/file.txt', 'lista3/data.txt', 'lista3/test_data.txt')
