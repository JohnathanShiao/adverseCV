from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import argparse
import cv2

def preprocess_image(image):
    #swap color channels, resize the input image, and add a batch dimension
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    image = np.expand_dims(image,axis=0)
    return image

def clip_eps(tensor,eps):
    #clip the values of the tensor to a given range and return it
    return tf.clip_by_value(tensor,clip_value_min=-eps,clip_value_max=eps)

def generate_targeted_adversaries(model,baseImage,delta,classIdx,target,steps=500):
    #iterate over the number of steps
    for step in range(0,steps):
        #record gradients
        with tf.GradientTape() as tape:
            #explicitly indicate that our perturbation vector should be tracked for gradient updates
            tape.watch(delta)

            #add our perturbation bector to the base image and preprocess the resulting image
            adversary = preprocess_input(baseImage + delta)

            #run this newly constructed image tensor through our model and calculate the loss with respect to both the original class and the target class labels
            predictions = model(adversary, training=False)
            originalLoss = -sccLoss(tf.convert_to_tensor([classIdx]), predictions)            
            targetLoss = sccLoss(tf.convert_to_tensor([target]), predictions)
            totalLoss = originalLoss + targetLoss

            #check to see if we are logging the loss value, and if so, display it to our terminal
            if step % 20 == 0:
                print("step: {}, loss: {}...".format(step,totalLoss.numpy()))

        #calculate the gradients of loss with respect tot the perturbation vector
        gradients = tape.gradient(totalLoss, delta)

        #update the weights, clip the perturbation vector, and update its value
        optimizer.apply_gradients([(gradients,delta)])
        delta.assign_add(clip_eps(delta,eps=EPS))
        
    #return the perturbation vector
    return delta

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True, help = "path to original input image")
ap.add_argument("-o","--output",required=True, help = "path to output adversarial image")
ap.add_argument("-c","--class-idx",type=int,required=True, help = "ImageNet clss ID of the predicted label")
ap.add_argument("-t","--target-class-idx",type=int,required=True, help = "ImageNet class ID of the target adversarial label")
args = vars(ap.parse_args())

EPS = 2 / 255.0
LR = 5e-3

#load image from disk and preprocess it
print("[INFO] loading image...")
image = cv2.imread(args["input"])
image = preprocess_image(image)

#load the pre-trained ResNet50 model for running inference
print("[INFO] loading pre-trained ResNet50 model...")
model = ResNet50(weights="imagenet")

#initialize optimizer and loss function
optimizer = Adam(learning_rate=LR)
sccLoss = SparseCategoricalCrossentropy()

#create a tensor based off the input image and initialie the perturbation vector (update via traning)
baseImage = tf.constant(image,dtype=tf.float32)
delta = tf.Variable(tf.zeros_like(baseImage),trainable=True)

#generate the perturbation vector to create an adversarial example
print("[INFO] generating perturbation...")
deltaUpdated = generate_targeted_adversaries(model, baseImage, delta, args["class_idx"], args["target_class_idx"])

#create the adversarial example, swap color channels, and save the output image to disk
print("[INFO] creating targeted adversarial example...")
adverImage = (baseImage + deltaUpdated).numpy().squeeze()
adverImage = np.clip(adverImage,0,255).astype("uint8")
adverImage = cv2.cvtColor(adverImage, cv2.COLOR_RGB2BGR)
cv2.imwrite(args["output"],adverImage)

#run inference with this adversarial example, parse the results, and display the top-1 predicted result
print("[INFO] running inference on the adversarial example...")
preprocessedImage = preprocess_input(baseImage+deltaUpdated)
predictions = model.predict(preprocessedImage)
predictions = decode_predictions(predictions,top=3)[0]
label = predictions[0][1]
confidence = predictions[0][2] * 100
print("[INFO] label: {} confidence: {:.2f}%".format(label,confidence))

#write the top-most predicted label on the image along with the confidence score
text = "{}: {:.2f}%".format(label,confidence)
cv2.putText(adverImage, text, (3,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

#show the output image
cv2.imshow("Output",adverImage)
cv2.waitKey(0)