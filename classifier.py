
"""
This class contains the code for training a random forest classifier to classify each pixel 
"""

def get_pixel_features(img, offsets):
    """
    This function returns a depth feature descriptor of the pixel 
    using the surrouding pixels and the feature response function 
    """
    # compute the feature response for each pixel 
    return []

def random_sample_offsets():
    """
    This function randomly samples offset values for the feature response function
    """
    n = 10 # number of offsets to sample, 
    offsets = []
    return offsets 

def train_random_forest_classifier(pixels, labels):
    """
    This function trains a random forest classfier using the sklearn ensemble library. It returns
    the model so that it can be used to classify the pixels of all images 
    """
    return 0

def classify_pixel(classifier, pixel_feature): 
    """
    This function classifies pixel as a body part using the trained random forest classifier.
    """
    return 0


#HIGHKEY put this in here cuz i have no clue how we are supposed to even do this
def get_pixel_labels(img): 
    """
    This function takes in an image and returns the pixel labels for each pixel in the image.
    """
    return 0

def estimate_joint_locations(pixel_dict): 
    """
    This function estimates the joint locations using the classified pixels for each body part and 
    returns an array of joint locations mapped by body part id   
    """
    return 0

def train(training_data): 
    """
    This function trains the random forest classifier using the training data 
    and returns the trained model. It takes in the training data and 
    """
    #for each image in the training set 
    # for each pixel in each image 
    # get the pixel features using the get_pixel_features function
    # get the pixel label using the get_pixel_labels function 
    # append to long list of pixels 
    # pass in said long list to train_random_forest_classifier function 
    # return the trained model 
    return 0

def classify_image(img): 
    """
    This function uses the trained model to classify all pixels of the input image and
    returns an array of estimated joint locations??? 
    """
    # for each pixel in image 
    # get the pixel features using the get_pixel_features function
    # classify the pixel using the classify_pixel function 
    # group each pixel by body part id dictionary {body_part_id: [pixel1, pixel2, ...]}
    # pass the classified pixels to the estimate_joint_locations function
    # return the estimated joint locations 
    return 0
