"""
Author: <Ravindu Santhush Ratnayake>

This code considers ANNs for the task of
recognising handwritten digits (0 to 9) from black-and-white images with a
resolution of 28x28 pixels. In assignment 1, you will create
functions that compute an ANN output for a given input

"""

def linear(x, w, b): # 1 Mark
    """
    Input: A list of inputs (x), a list of weights (w) and a bias (b).
    Output: A single number corresponding to the value of f(x) in Equation 1.

    >>> x = [1.0, 3.5]
    >>> w = [3.8, 1.5]
    >>> b = -1.7
    >>> round(linear(x, w, b),6) #linear(x, w, b)
    7.35
    """
    functionOutput = 0
    for index in range(len(x)):
        functionOutput += x[index] * w[index] 
    functionOutput += + b 
    return(functionOutput)
    pass


def linear_layer(x, w, b): # 1 Mark
    """
    Input: A list of inputs (x), a table of weights (w) and a list of 
           biases (b).
    Output: A list of numbers corresponding to the values of f(x) in
            Equation 2.
    
    >>> x = [1.0, 3.5]
    >>> w = [[3.8, 1.5], [-1.2, 1.1]]
    >>> b = [-1.7, 2.5]
    >>> y = linear_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #linear_layer(x, w, b)
    [7.35, 5.15]
    """
    outputLayer = []
    for index in range(len(w)):
        outputLayer.append(linear(x, w[index], b[index]))
    return(outputLayer)
    pass


def inner_layer(x, w, b): # 1 Mark
    """
    Input: A list of inputs (x), a table of weights (w) and a 
           list of biases (b).
    Output: A list of numbers corresponding to the values of f(x) in 
            Equation 4.

    >>> x = [1, 0]
    >>> w = [[2.1, -3.1], [-0.7, 4.1]]
    >>> b = [-1.1, 4.2]
    >>> y = inner_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inner_layer(x, w, b)
    [1.0, 3.5]
    >>> x = [0, 1]
    >>> y = inner_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inner_layer(x, w, b)
    [0.0, 8.3]
    """
    outputLayer = []
    for index in range(len(w)):
        outputLayer.append(max(linear(x, w[index], b[index]), 0.0))
    return(outputLayer)
    pass


def inference(x, w, b): # 2 Marks
    """
    Input: A list of inputs (x), a list of tables of weights (w) and a table
           of biases (b).
    Output: A list of numbers corresponding to output of the ANN.
    
    >>> x = [1, 0]
    >>> w = [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> b = [[-1.1, 4.2], [-1.7, 2.5]]
    >>> y = inference(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inference(x, w, b)
    [7.35, 5.15]
    """
    for index in range(len(w)):
        x = inner_layer(x, w[index], b[index])
    return(x)
    pass

def read_weights_biases(filename): # 2 Marks
    """
    Input: A string (filename), that corresponds to the name of the file that contains the weights and biases of the ANN.

    Output: A tuple w, b corresponding to the weights and biases of the ANN. 
    
    For example:

    >>> w_example, b_example = read_weights_biases('example_weights_biases.txt')
    >>> w_example
    [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> b_example
    [[-1.1, 4.2], [-1.7, 2.5]]
    """
    biasesList = []
    weightsList =[]
    biasesSubList = []
    weightsSubList = []
    dataList=[]
    
    myFile = open(filename, 'r')
    for line in myFile:
        dataList.append(line.strip())  
    myFile.close()

    for index in range(len(dataList)):
        if ((index + 1) < len(dataList)):
            if(dataList[index] == "#w"):            
                while(dataList[index + 1] != "#b"):                
                    index += 1
                    splitList = dataList[index].strip().split(",")
                    splitList = [float(x) for x in splitList]
                    weightsSubList.append(splitList)
                    if((index + 1) == len(dataList)):
                        break
                weightsList.append(weightsSubList)
                weightsSubList=[]
       
            
            if(dataList[index] == "#b"):            
                while(dataList[index + 1] != "#w"):                
                    index += 1
                    splitList = dataList[index].strip().split(",")
                    splitList = [float(x) for x in splitList]
                    biasesSubList =biasesSubList+(splitList)
                    if((index + 1) == len(dataList)):
                        break
                biasesList.append(biasesSubList)
                biasesSubList=[]
    return((weightsList), (biasesList))
    pass


def read_image(file_name): # 1 Mark
    """
    Input: A string (file_name), that corresponds to the name of the file
           that contains the image.
    Output: A list of numbers corresponding to input of the ANN.
    
    >>> x = read_image('image.txt')
    >>> len(x)
    784
    """
    myFile  = open(file_name, 'r')
    outputList =[]
    for line in myFile:
        line = line.strip()
        outputList = outputList + [int(binaryNum) for binaryNum in line]
    return(outputList)
    pass


def argmax(x): # 1 Mark
    """
    Input: A list of numbers (i.e., x) that can represent the scores 
           computed by the ANN.
    Output: A number representing the index of an element with the maximum
            value, the function should return the minimum index.
    
    >>> x = [1.3, -1.52, 3.9, 0.1, 3.9]
    >>> argmax(x)
    2
    """
    highestIndex = 0
    for index in range(len(x)):
        if(x[index] > x[highestIndex]):
            highestIndex = index
    return(highestIndex)
    pass


def predict_number(image_file_name, weights_biases_file_name): # 1 Mark
    """
    Input: A string (i.e., image_file_name) that corresponds to the image
           file name, a string (i.e., weights_biases_file_name) that corresponds
           to the weights and biases file name.
    Output: The number predicted in the image by the ANN.

    >>> i = predict_number('image.txt', 'weights_biases.txt')
    >>> print('The image is number ' + str(i))
    The image is number 4
    """
    imageFile = read_image(image_file_name)
    weightsList, biasesList = read_weights_biases(weights_biases_file_name)
    
    outputLayer = inference(imageFile, weightsList, biasesList)
    predictedNum = argmax(outputLayer)
     
    return(str(predictedNum))
    pass

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
