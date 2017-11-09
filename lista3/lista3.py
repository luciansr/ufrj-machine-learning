import random


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

# randomizeFile('lista3/file.txt', 'lista3/data.txt', 'lista3/test_data.txt')