
import numpy as np
from matplotlib import pyplot as plt

class FLMetric()
    def __init__(self):
        self._metrics = {}

    def add_metric(self, name, metric):
        self._metrics[name] = metric
    
    @staticmethod
    def calc_tp_fp_tn_fn(correct_labels_in,predicted_labels_in, num_classes):
        correct_labels, predicted_labels = np.array(correct_labels_in), np.array(predicted_labels_in)

        TP, FP, TN, FN = [], [], [], []

        for i in range(num_classes):    
            TP.append( (((correct_labels==i).astype(int) + (predicted_labels==i).astype(int)) == 2).sum().item() )
            FP.append( (((correct_labels!=i).astype(int) + (predicted_labels==i).astype(int)) == 2).sum().item() )
            TN.append( (((correct_labels!=i).astype(int) + (predicted_labels!=i).astype(int)) == 2).sum().item() )
            FN.append( (((correct_labels==i).astype(int) + (predicted_labels!=i).astype(int)) == 2).sum().item() )

        return np.array(TP), np.array(FP), np.array(TN), np.array(FN)

    @staticmethod
    def calc_precision(correct_labels,predicted_labels,num_classes):
        TP, FP, _, _ = Metrics.calc_tp_fp_tn_fn(correct_labels, predicted_labels, num_classes)

        precision = []
        
        for i in range(num_classes):
            try:
                precision.append( TP[i]/(TP[i]+FP[i]) )
            except (ZeroDivisionError, FloatingPointError):
                precision.append( 0.0 )
        
        return np.array(precision)

    @staticmethod
    def calc_recall(correct_labels,predicted_labels,num_classes):
        TP, _, _, FN = Metrics.calc_tp_fp_tn_fn(correct_labels, predicted_labels, num_classes)

        recall = []

        for i in range(num_classes):
            try:
                recall.append( TP[i]/(TP[i]+FN[i]) )
            except (ZeroDivisionError, FloatingPointError):
                recall.append( 0.0 )

        return np.array(recall)

    @staticmethod
    def calc_specificity(correct_labels,predicted_labels,num_classes):
        _, FP, TN, _ = Metrics.calc_tp_fp_tn_fn(correct_labels, predicted_labels, num_classes)

        specificity = []

        for i in range(num_classes):
            try:
                specificity.append( TN[i]/(TN[i]+FP[i]) )
            except (ZeroDivisionError, FloatingPointError):
                specificity.append( 0.0 )
        
        return np.array(specificity)

    @staticmethod
    def calc_npv(correct_labels, predicted_labels, num_classes):
        _, _, TN, FN = Metrics.calc_tp_fp_tn_fn(correct_labels, predicted_labels, num_classes)

        npv = []

        for i in range(num_classes):
            try:
                npv.append( TN[i]/(TN[i]+FN[i]) )
            except (ZeroDivisionError, FloatingPointError):
                npv.append( 0.0 )
        
        return np.array(npv)

    @staticmethod
    def calc_f1(correct_labels, predicted_labels, num_classes):
        precision = Metrics.calc_precision(correct_labels, predicted_labels, num_classes)
        recall = Metrics.calc_recall(correct_labels, predicted_labels, num_classes)

        precision_mean = precision[1:].mean() if len(precision) > 0 else 0.0
        recall_mean = recall[1:].mean() if len(recall) > 0 else 0.0

        try:
            f1score = 2*(precision_mean*recall_mean) / (precision_mean+recall_mean)
        except (ZeroDivisionError, FloatingPointError):
            f1score = 0.0
        
        return f1score

    @staticmethod
    def calc_confusionMatrix(correct_labels_in, predicted_labels_in, num_classes):
        correct_labels, predicted_labels = np.array(correct_labels_in), np.array(predicted_labels_in)

        confusionMatrix_array = metrics.confusion_matrix(correct_labels, predicted_labels, labels=list(range(num_classes)))

        return confusionMatrix_array

    @staticmethod
    def calc_accuracy_classification(correct_labels, predicted_labels, num_classes):
        accuracy = []
        for i in range(num_classes):
            predicted_labels_class = [item for j,item in enumerate(predicted_labels) if correct_labels[j]==i]
            correct_labels_class = [item for j,item in enumerate(correct_labels) if correct_labels[j]==i]

            TP, FP, TN, FN = Metrics.calc_tp_fp_tn_fn(correct_labels_class, predicted_labels_class, 1)

            correct = TP + TN
            total = TP + FP + TN + FN
            try:
                accuracy.extend(correct / total)
            except (ZeroDivisionError, FloatingPointError):
                accuracy.append(0.0)

        return np.array(accuracy)

    @staticmethod
    def calc_accuracy_balanced_classification(correct_labels, predicted_labels, num_classes): # macro-average
        recall = Metrics.calc_recall(correct_labels, predicted_labels, num_classes)
        specificity = Metrics.calc_specificity(correct_labels, predicted_labels, num_classes)
        
        recall_mean = recall[1:].mean() if len(recall) > 0 else 0.0
        specificity_mean = specificity[1:].mean() if len(specificity) > 0 else 0.0

        accuracy_balanced = (recall_mean+specificity_mean) / 2

        return accuracy_balanced

    @staticmethod
    def plot_confusionMatrix(confusionMatrix_array, classes, title='', normalize=False, cmap=plt.cm.Greens, closeFigure=False):
        """
        This function plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        plt.ioff()
        fig, ax = plt.subplots( figsize=(2*(len(classes)-1), 2*(len(classes)-1)) )
        img = ax.imshow(confusionMatrix_array, interpolation='nearest', cmap=cmap)
        plt.colorbar(img, ax=ax)
        ax.title.set_text(title)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        if normalize:
            confusionMatrix_array = confusionMatrix_array.astype('float') / confusionMatrix_array.sum(axis=1)[:, np.newaxis]

        thresh = confusionMatrix_array.max() / 2.
        for i, j in itertools.product(range(confusionMatrix_array.shape[0]), range(confusionMatrix_array.shape[1])):
            ax.text(j, i, confusionMatrix_array[i, j],
                    horizontalalignment="center",
                    color="white" if confusionMatrix_array[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', dpi=100)
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = torchvision.transforms.ToTensor()(image)
        plt.clf()

        if closeFigure:
            plt.close(fig)
            return image
        else:
            return fig, image