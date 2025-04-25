
import numpy as np 
"""
This class contains the code for training a random forest classifier to classify each pixel 
"""

def get_feature_vector(img, offsets, joints):
    """
    This function returns a depth feature descriptor of each pixel in a joint vector  
    using the surrouding pixels and the feature response function. it concatenates each pixel 
    descriptor into a vector of num_joints x num_offsets x 2 (15 x 100 x 2)
    """
    # compute the feature response for each pixel 
    joints = np.array(joints)
    ft_vector = np.zeros((joints.shape[0], len(offsets)))
    for i, joint in enumerate(joints): 
        feature = np.zeros(len(offsets))
        x, y = joint
        d = img[x][y]
        for j, offset in enumerate(offsets): 
            delta_x, delta_y = offset
            feature[j] = d - img[x + delta_x, y + delta_y]
        ft_vector[i] = feature 
    return ft_vector

def random_sample_offsets(max_x, max_y):
    """
    This function randomly samples offset values for the feature response function
    """
    num_offsets = 100 # number of offsets to sample, 
    offset_threshold = 30 # highest offset value 
    offsets = []
    for _ in range(num_offsets): 
        x = np.random.randint(-1 * (offset_threshold + 1), offset_threshold + 1)
        y = np.random.randint(-1 * (offset_threshold + 1), offset_threshold + 1)
        offsets.append((x, y))
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
def get_pixel_labels(data, labels): 
    """
    This function takes in an image and returns the pixel labels for each pixel in the image. It uses 
    the joint position given by the label to recover the pixel labels. 
    """
    return 0

def estimate_joint_locations(pixel_dict): 
    """
    This function estimates the joint locations using the classified pixels for each body part and 
    returns an array of joint locations mapped by body part id   
    """
    return 0

def train(train_data, train_labels): 
    """
    This function trains the random forest classifier using the training data 
    and returns the trained model. It takes in the training data and training labels
    """
    #for each image in the training set 
    # for each pixel in each image 
    # get the pixel features using the get_pixel_features function
    # get pixel labels using the get_pixel_labels function 
    # append to long list of pixels 
    # pass in said long list to train_random_forest_classifier function 
    # return the trained model 
    return 0

def test(test_data, test_labels): 
    # test each image in the test set 
    # calculate difference between ground truth and prediction 
    # print accuracy 
    return 

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


