# Standard Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os

from PIL import Image

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score

# Lime Imports
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Compare normal x-ray with pneumonia x-ray, randomly selected from training data
def compare_xrays():
    train_path = '../../chest_xray/train'
    normal_file = random.choice(os.listdir(train_path + '/NORMAL'))
    normal_path = os.path.join(train_path + '/NORMAL', normal_file)
    normal_img = mpimg.imread(normal_path)
    pneumonia_file = random.choice(os.listdir(train_path + '/PNEUMONIA'))
    pneumonia_path = os.path.join(train_path + '/PNEUMONIA', pneumonia_file)
    pneumonia_img = mpimg.imread(pneumonia_path)

    plt.figure(figsize = (20, 8))
    ax = plt.subplot(1, 2, 1)
    ax.set_title('Normal X-Ray', fontdict={'size': 18})
    plt.imshow(normal_img)
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Pneumonia X-Ray', fontdict = {'size': 18})
    plt.imshow(pneumonia_img);
    plt.savefig('../../reports/figures/x-ray_compare')
    return


# Function to plot loss, accuracy and recall during training
def visualize_training_results(results):
    """Input results = model.fit
    requires both accuracy and recall as metrics
     
    """
    history = results.history
    plt.figure()
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('../../reports/figures/loss_results')
    plt.show()
    
    plt.figure()
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('../../reports/figures/accuracy_results')
    plt.show()
    
    plt.figure()
    plt.plot(history['val_recall'])
    plt.plot(history['recall'])
    plt.legend(['val_recall', 'recall'])
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.savefig('../../reports/figures/recall_results')
    plt.show();
    return

def create_confusion_matrix(model, generator):
    """Input model and generator
    Creates confusion matrix     
    """
    preds = (model.predict(generator) > 0.5).astype('int32')
    true_labels = generator.classes
    labels = list(generator.class_indices.keys())
    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
    fig, ax = plt.subplots(figsize = (8, 8))
    disp.plot(ax = ax)
    plt.grid(False)
    plt.savefig('../../reports/figures/confusion_matrix')
    plt.show();
    return

def get_metrics(model, generator):
    """Input model and generator
    Prints accuracy and recall     
    """
    preds = (model.predict(generator) > 0.5).astype('int32')
    true_labels = generator.classes
    print('Accuracy:', accuracy_score(true_labels, preds))
    print('Recall:', recall_score(true_labels, preds))
    return

def display_lime(model, generator):
    """Input model and generator
    Displays Lime for 3 random x-rays selected from generator using model
    """
    explainer = lime_image.LimeImageExplainer()
    plt.figure(figsize = (12, 10))
    for i in range(3):
        # Randomly select image in generator
        batch_no = random.choice(range(len(generator[0][0])))
        ax = plt.subplot(1, 3, i+1)
        explanation = explainer.explain_instance(generator[0][0][batch_no].astype('double'), 
                                                 model.predict, top_labels = 1,
                                                 hide_color = 0, num_samples = 1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                    positive_only = False, num_features = 10, 
                                                    hide_rest = False)
        if generator[0][1][batch_no] == 0:
            ax.set_title('Normal X-Ray', fontdict={'size': 18})
        else:
            ax.set_title('Pneumonia X-Ray', fontdict={'size': 18})
        plt.imshow(mark_boundaries(temp, mask))
    plt.savefig('../../reports/figures/lime_figure')
    plt.show();
    return

def run_lime_heatmap(model, generator):
    """Input model and generator
    Displays X-Ray and Lime Heatmap for random x-ray selected from generator using model     
    """
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (20, 10))
    explainer = lime_image.LimeImageExplainer()

    # Randomly select image in generator
    batch_no = random.choice(range(len(generator[0][0])))
    explanation = explainer.explain_instance(generator[0][0][batch_no].astype('double'), 
                                             model.predict, top_labels = 2,
                                             hide_color = 0, num_samples = 1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                positive_only = False, num_features = 1000, 
                                                hide_rest = False, min_weight = 0.1)
    if generator[0][1][batch_no] == 0:
        ax1.set_title('Normal X-Ray', fontdict={'size': 18})
    else:
        ax1.set_title('Pneumonia X-Ray', fontdict={'size': 18})
    ax1.imshow(generator[0][0][batch_no])

    ind = explanation.top_labels[0]
    
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    
    # Plot.  The visualization makes more sense if a symmetrical colorbar is used
    ax2.set_title('Lime Heatmap', fontdict={'size': 18})
    img = ax2.imshow(heatmap, cmap = 'RdBu', vmin = -heatmap.max(), vmax = heatmap.max())
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(img, cax = cax)
    plt.savefig('../../reports/figures/heatmap')
    plt.show()
    return



def run_three_lime_heatmaps(model, generator):
    """Input model and generator
    Displays Lime Heatmap for 3 random x-rays selected from generator using model
    """
    explainer = lime_image.LimeImageExplainer()
    plt.figure(figsize = (12, 10))
    for i in range(3):
        # Randomly select image in generator
        batch_no = random.choice(range(len(generator[0][0])))
        explanation = explainer.explain_instance(generator[0][0][batch_no].astype('double'), 
                                                 model.predict, top_labels = 2,
                                                 hide_color = 0, num_samples = 1000)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                    positive_only = False, num_features = 1000, 
                                                    hide_rest = False, min_weight = 0.1)

        # Select the same class explained in the figures above
        ind = explanation.top_labels[0]
    
        # Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    
        # Plot.  The visualization makes more sense if a symmetrical colorbar is used
        ax = plt.subplot(1, 3, i+1)
        if generator[0][1][batch_no] == 0:
            ax.set_title('Normal X-Ray', fontdict={'size': 18})
        else:
            ax.set_title('Pneumonia X-Ray', fontdict={'size': 18})
        plt.imshow(heatmap, cmap = 'RdBu', vmin = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()        
    plt.savefig('../../reports/figures/lime_heatmap')
    plt.show();
    return
