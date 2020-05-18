import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import keras
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D
from keras import backend as K

random.seed(a=None, version=2)

set_verbosity(INFO)

def get_mean_std_per_batch(image_path, df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        # path = image_dir + img
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std


def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(img_path, df, H=H, W=W)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_gradcam(model, img, image_dir, df, labels, selected_labels,
                    layer_name='bn'):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            # print("Loading original image")
            plt.figure(figsize=(15, 10))
            # plt.subplot(151)
            # plt.title("Original")
            # plt.axis('off')
            # plt.imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            # plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, df, preprocess=False),
                       cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            plt.savefig("static/anomaly_images/"+str(labels[i]+".jpg"))
            j += 1


def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals

anomaly_labels = ['Cardiomegaly', 
		          'Emphysema', 
		          'Effusion', 
		          'Hernia', 
		          'Infiltration', 
		          'Mass', 
		          'Nodule', 
		          'Atelectasis',
		          'Pneumothorax',
		          'Pleural_Thickening', 
		          'Pneumonia', 
		          'Fibrosis', 
		          'Edema', 
		          'Consolidation']

pos_weights = [0.02,0.013,0.128, 0.002, 0.175, 0.045, 0.054, 0.106, 0.038, 0.021, 0.01,  0.014, 0.016, 0.033]

neg_weights = [0.98 , 0.987 ,0.872, 0.998, 0.825, 0.955, 0.946, 0.894, 0.962, 0.979, 0.99,  0.986, 0.984, 0.967]
 
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):

    def weighted_loss(y_true, y_pred):

        loss = 0.0
        
        for i in range(len(pos_weights)):
            loss += -K.mean(pos_weights *y_true * K.log(y_pred + epsilon)+ neg_weights*(1-y_true)* K.log(1-y_pred + epsilon)) #complete this line
        return loss
    
    return weighted_loss

def detect_anomaly(image_filename):

	base_model = DenseNet121(weights='densenet.hdf5', include_top=False)

	x = base_model.output

	x = GlobalAveragePooling2D()(x)

	predictions = keras.layers.Dense(len(anomaly_labels), activation="sigmoid")(x)

	model = keras.models.Model(inputs=base_model.input, outputs=predictions)
	model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights))


	model.load_weights("anomaly_detection_pretrained_model.h5")


	im = cv2.resize(cv2.imread("static/"+image_filename), (224, 224)).astype(np.float32)
	im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
	im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
	im[:,:,2] = (im[:,:,2] - 123.68) * 0.017

	im = np.expand_dims(im, axis=0)

	out = model.predict(im)

	# print('Prediction: '+str(anomaly_labels[np.argmax(out)]))
	# print(out)

	# print(out[0][1])

	# for i in range(len(anomaly_labels)-1):
	    # print(anomaly_labels[i],out[0][i])    

	df = pd.read_csv("anomaly-detection-train-small.csv")

	labels_to_show = []
	res = {anomaly_labels[i]: out[0][i] for i in range(len(anomaly_labels))} 
	res_sorted =  sorted(res,key = res.get, reverse = True)
	answer = ""
	for itr in range(len(res)):
	  if res[anomaly_labels[itr]] > 0.85:
	    labels_to_show.append(anomaly_labels[itr])
	    answer += str(anomaly_labels[itr]) + " : " + str(res[anomaly_labels[itr]]) + " | "

	answer = answer[:-3]
	    
	compute_gradcam(model, "static/"+image_filename, image_dir="", df=df, labels=anomaly_labels, selected_labels=labels_to_show)
	plt.show()

	return answer