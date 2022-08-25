"""
Author: <Ravindu Santhush Ratnayake>

This code onsiders ANNs for the taks of
recognising handwritten digits (0 to 9) from black-and-white images with a
resolution of 28x28 pixels.
In assignment 2, you will write functions to "attack" an ANN. That is, to try to generate inputs
that will fool the network to make a wrong classification.

"""
from A1_solution import read_weights_biases , read_image, inference, argmax, predict_number
def compute_difference(x, y):
    # to compute the how many pixels are different between two images 
    return sum(1 if i !=j else 0 for i,j in zip(x,y))
def print_image(x):
    # to print a given image 
    c = 0
    for i, pixel in enumerate(x):
        if i%28 == 0:
            print("")
        else: 
            print(pixel, end = '')

# the variables were named this way as it was the names given in the assignment brief        
def select_pixel(x, w, b):
    """
    Input: A list of inputs (x), a list of tables of weights (w) and a table of biases (b).

    Output: An integer (i) either representing the pixel that is selected to be flipped, or with value -1 representing no further modifications can be made.

    For example given in image.txt, the function select_pixel(x, w, b) can behave as follows for input  list x, list of tables of weights
    w and table  of biases b:

    >>> x , (w, b) = read_image('image.txt'), read_weights_biases('weights_biases.txt')
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    238
    >>> x[pixel] = int(not x[pixel])
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    210

    For example given in another_image.txt, the function select_pixel(x, w, b) can behave as follows for input  list x, list of tables of weights
    w and table  of biases b:

    >>> x , (w, b) = read_image('another_image.txt'), read_weights_biases('weights_biases.txt')
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    343
    >>> x[pixel] = int(not x[pixel])
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    469
    """
    #here the intial part is to generate the inference list
    inferencedList = inference(x, w, b)
    #now i get the index value of the index of highest prediction
    originalConfindenceValueIndex = argmax(inferencedList)
    #store the orignal confidence value for later use
    #i have to pop the value off as i need to get the second highest value as well
    originalConfindenceValue = inferencedList.pop(originalConfindenceValueIndex)
    
    #now i get the index value of the highest index
    originalSecondConfindenceValueIndex = argmax(inferencedList)
    
    #here i store the actual value of the second highest value
    originalSecondConfindenceValue =  inferencedList.pop(originalSecondConfindenceValueIndex)
   
    
    #this is the impact list that will be used to store the impact value for all the changed pixels
    imapcLevelList = []

    #iterate through the image list that is inserted 
    for index in range(len(x)):
        #invert the pixel of the current index
        x[index] = int(not x[index])
        #generate the new confidence list
        perturbedConfidenceList = inference(x, w, b)
        #anow i get the NEW confidence value of the original confidence value index
        perturbedHighestConfidenceVal = perturbedConfidenceList[originalConfindenceValueIndex]

        #similarly i get the NEW confidence value of the second original confidence value index
        perturbed2ndHighestConfidenceVal = perturbedConfidenceList[originalSecondConfindenceValueIndex]
        
        #this is the point where i calculate the the impact values
        #it was stated that if the original confidence value was lowered take it as positive
        #therefore if the changed confidence value is larger when substracted it will be a negative value
        changeOfOriginalGuess = originalConfindenceValue - perturbedHighestConfidenceVal

        
        #as for the second highest value if the new confidence value is larger it must be considered positive
        #therefore if the new confidence value is larger if substracted  from the original the result will be positive
        changeOf2ndOriginalGuess = perturbed2ndHighestConfidenceVal - originalSecondConfindenceValue

        #here i add the two values up
        impact = changeOfOriginalGuess + changeOf2ndOriginalGuess
        #i then append the impact value to the list
        imapcLevelList.append(impact)


        #finally the pixel is again inverted to its original value
        x[index] = int(not x[index])

    
    #now i get the max vlaue from the impact level list
    highestConfidence =  argmax(imapcLevelList)
    #i check if the impact is less than zero
    #if it is less than zero it means that the maximum impact has increased the prediction confidence of the original prediction
    if highestConfidence < 0:
        return -1
    #if a positive impact has been made we return the highest confidence index
    else:
        return(highestConfidence)   

    
        
    
    
    
    pass

def adversarial_image(image_file_name,weights_biases_file_name):
    '''
    >>> file_name = 'image.txt'
    >>> orig_image, adv_image = read_image(file_name), adversarial_image(file_name, 'weights_biases.txt')
    >>> if adv_image == -1:
    ...     print('Algorithm failed.')
    ... else:
    ...     print('An adversarial image is found!, with ', compute_difference(orig_image, adv_image), ' pixels change')
    ...     print_image(adv_image)
    An adversarial image is found!, with  2  pixels change
    <BLANKLINE>
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000110010000110000000
    000000000110010000110000000
    000000001100000000110000000
    000000011000000000110000000
    000000011000000001110000000
    000000110000000001100000000
    000000110000000011100000000
    000000110000000011100000000
    000000110000000011000000000
    000000111000011111000000000
    000000011111111111000000000
    000000000000000111000000000
    000000000000000011000000000
    000000000000000111000000000
    000000000000000011000000000
    000000000000000111000000000
    000000000000000110000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000

    >>> file_name = 'another_image.txt'
    >>> orig_image, adv_image = read_image(file_name), adversarial_image(file_name, 'weights_biases.txt')
    >>> if adv_image == -1:
    ...    print('Algorithm failed.')
    ... else:
    ...    print('An adversarial image is found!, with ', compute_difference(orig_image, adv_image), ' pixels change')
    ...    print_image(adv_image)
    An adversarial image is found!, with  2  pixels change
    <BLANKLINE>
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000011111111110000000
    000000001111111111111000000
    000000011100000000111000000
    000000111000000000111000000
    000000111000000000111000000
    000000111000000001111000000
    000000011000000001110000000
    000000111111111111100000000
    000000011111111111100000000
    000000111111111111110000000
    000001111100000111111000000
    000001110000000001110000000
    000001110000000001110000000
    000001110000000001110000000
    000001111000000111100000000
    000000111111111111000000000
    000000011111111100000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    000000000000000000000000000
    '''
    #begining off with this function i use the previous assignment functions to read the image file and biases file
    #originalImage = read_image(image_file_name)
    imageFile = read_image(image_file_name)
    weightList, biasesList = read_weights_biases("weights_biases.txt")

    #here i get the first prediction before any changes are done to the image file
    firstPrediction = argmax(inference(imageFile, weightList, biasesList))

    #i now make a new variable to use in a while loop to check if the prediction has changed
    currentPrediction = firstPrediction
    

    #this is used to keep track of how many pixels have been changed    
    numPurbations = 0
    #if the amount of pixels changed is equal to the number of pixels then i will use it to exit the loop
    imgFileLength = len(imageFile)

    #this loo[ will run until the slect pixel function returens -1
    #until number of all pixels changed are equal to the number of total pixels plus 1
    #or till the prediction is changed    
    while(currentPrediction == firstPrediction):
        # get the pxel that is to be changed using select pixel function
        perturbedPixel = select_pixel(imageFile, weightList, biasesList)
        #if -1 is returned we return -1 
        if perturbedPixel == -1:
            return(-1)
        #if a pixel is selcted then we change the image
        imageFile[perturbedPixel] = int(not imageFile[perturbedPixel])

        #increment the number of changed pixels
        numPurbations += 1
        #check if more pixels than the number of pixels are changed if true then we return -1
        if numPurbations == imgFileLength + 1:
            return(-1)
        #chenage the current prediction according to the newly changed image
        currentPrediction = argmax(inference(imageFile, weightList, biasesList))

    #return the changed image file as required
    return(imageFile)
    #below commands used to test if the file is working properly
    #print('An adversarial image is found!, with ', compute_difference(originalImage, imageFile), ' pixels change')
    #print_image(imageFile)
    pass


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

