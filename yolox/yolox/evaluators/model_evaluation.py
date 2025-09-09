from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import json
import os
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from loguru import logger

class EvaluateModel:
    
    def __init__(self):
        pass 
    
    
    def get_precision(self, TP, FP):
        precision = TP / (TP + FP) 
        return precision
    
    
    def get_recall(self, TP, FN):
        recall = TP / (TP + FN)
        return recall
        
    
    def get_f1_score(self, precision, recall):
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def get_mAP(self):
        pass
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        Each box should be a list in the format [x1, y1, x2, y2].
        (x1, y1) represents the top-left coordinate,
        and (x2, y2) represents the bottom-right coordinate.
        """
        iou = maskUtils.iou([box1], [box2], [1]) #isCrowd 1 because results look more correct than 0 (non sense)

        return iou
    
    
    def evaluate(self, ground_truth_json, predictions_json, iou_threshold=0.5, threshold=0.5):
        
        # ground_truth_json and predictions_json must be in coco format
        # coco format for gound_truth_json can be downladed using Label Studio
        
        # evaluation for the predictions of object detection models
        # each GT bounding box is matched with a predicted one according to the iou
        # duplicated boxes are considered false positives
        # a bounding box is assigned only once 
        
        # Load ground truth and predictions
        if isinstance(ground_truth_json, COCO):
            ground_truth = ground_truth_json
        else:
            ground_truth = COCO(ground_truth_json)
        classes = ground_truth.getCatIds()
        predictions = ground_truth.loadRes(predictions_json)  
        
        # Get image IDs
        image_ids = ground_truth.getImgIds()
        
        dict_summary = {}
        dict_summary["all"] = []

        # Iterate over image IDs
        for image_id in tqdm(image_ids, desc="Processing evaluation:"):
            
            # Load annotations for the current image
            ground_truth_anns = ground_truth.loadAnns(ground_truth.getAnnIds(imgIds=image_id))
            prediction_anns = predictions.loadAnns(predictions.getAnnIds(imgIds=image_id))
            
            # Calculate IoU between ground truth and prediction bounding boxes            
            matrix_iou = np.zeros((len(ground_truth_anns), len(prediction_anns)))
            matrix_scores = np.zeros((len(ground_truth_anns), len(prediction_anns)))
            # Confusion matrix: GT columns, pred rows
            # It is possible to have more predictions than GT bboxes and the sum of all values are greater than the number of GT bboxes
            confusion_matrix_im = np.zeros((len(classes)+1, len(classes)+1)) #+1 to add background category
            for i, gt_ann in enumerate(ground_truth_anns):
                gt_bbox = gt_ann['bbox']
                #gt_mask = ground_truth.annToMask(gt_ann)
                for j, pred_ann in enumerate(prediction_anns):
                    pred_bbox = pred_ann['bbox']
                    #pred_mask = predictions.annToMask(pred_ann)
                    iou = self.calculate_iou(gt_bbox, pred_bbox)
                    matrix_iou[i,j] = iou
                    matrix_scores[i,j] = pred_ann['score'] #store scores to filter with a threshold
                    
            row_ind, col_ind = linear_sum_assignment(-matrix_iou) #hungarian matrix (negative to look for the minimum)
            
            assigned_index = []
            for row, col in zip(row_ind, col_ind):
                
                if matrix_scores[row, col] >= threshold and matrix_iou[row, col] >= iou_threshold:
                    gt_class = ground_truth_anns[row]['category_id'] # ground truth class
                    pred_class = prediction_anns[col]['category_id'] # predicted class
                    confusion_matrix_im[gt_class,pred_class] += 1  # add the category to the confusion matrix
                    assigned_index.append(col)
                    
                
                        
            '''assigned_index = []
            for row in range(0, matrix_iou.shape[0]):
                 
                 indexed_list = list(enumerate(matrix_iou[row]))
                 # Sort the list in descending order based on the second element of each tuple (the actual value)
                 sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
                 
                 for best_iou_row, iou_value in sorted_list:
                    best_iou_column = np.argmax(matrix_iou[:,best_iou_row])
                    if best_iou_column == row and best_iou_row not in assigned_index and iou_value > iou_threshold and matrix_scores[row, best_iou_row] >= threshold:
                        assigned_index.append(best_iou_row)
                        gt_class = ground_truth_anns[row]['category_id'] # ground truth class
                        pred_class = prediction_anns[best_iou_row]['category_id'] # predicted class
                        confusion_matrix_im[gt_class,pred_class] += 1  # add the category to the confusion matrix
                        break'''
                     
            # Look for false positives and false negatives for the bboxes not matched
            # False negatives
            FN = []
            for i in range(len(ground_truth_anns)):
                if i not in assigned_index:
                    FN.append(i)
                    gt_class = ground_truth_anns[i]['category_id']
                    confusion_matrix_im[-1,gt_class] += 1 
            
            #False positives      
            for j in range(len(prediction_anns)):
                score = prediction_anns[j]['score']
                if j not in assigned_index and i not in FN and score >= threshold:
                    pred_class = prediction_anns[j]['category_id']
                    confusion_matrix_im[pred_class,-1] += 1 
                    
            image_summary = {"image_id": image_id, "total_gt_boxes": len(ground_truth_anns),
                             "confusion_matrix": confusion_matrix_im} 
            
            
            dict_summary["all"].append(image_summary)
            
        return dict_summary

    
    def accumulate(self, dict_summary):
        
        dict_summary_all = {}
        
        conf_matrix_all = np.zeros(dict_summary["all"][0]["confusion_matrix"].shape)
        
        # Sum all confusion matrix from all images
        for data in dict_summary["all"]:
            
            confusion_matrix = data["confusion_matrix"]
            conf_matrix_all += confusion_matrix
                
        # Precision, recall and f1_score for each class except background class       
        for row in range(0, len(conf_matrix_all)-1): #-1 background is not computed
            
            TP = conf_matrix_all[row,row]
            FP = np.sum(conf_matrix_all[row, :]) - TP # just we sum the rest of the classes
            FN = np.sum(conf_matrix_all[:, row]) - TP # just we sum the rest of the classes
            
            precision = self.get_precision(TP, FP)
            recall = self.get_recall(TP, FN)
            f1_score = self.get_f1_score(precision, recall)
            dict_summary_all["category_%s"%(str(row))] = {"precision": precision, "recall": recall, "f1_score": f1_score}       

        dict_summary_all["conf_matrix_all"] = conf_matrix_all.tolist()
    
        return dict_summary_all
    
    def summary(self, dict_summary_all, output_filename=None):
        
        #output filename in json format
        
        keys_except_last = list(dict_summary_all.keys())[:-1] #last one is conf_matrix_all
        
        for row in keys_except_last:
            logger.info("Precision %s: "%(row), dict_summary_all[row]["precision"])
            logger.info("Recall %s: "%(row), dict_summary_all[row]["recall"])
            logger.info("f1_score %s: "%(row), dict_summary_all[row]["f1_score"])
            print("----------------------------------------------------------")
        
        extension = output_filename.split(".")[-1]
        
        if output_filename != None:
            if extension == "json":
                with open(output_filename, "w") as json_file:
                    json.dump(dict_summary_all, json_file)
                
                json_file.close()
            
            else:
                return dict_summary_all
        
        

if __name__ == '__main__':

    evaluation = EvaluateModel()

    ground_truth_file = os.path.dirname(os.path.abspath(__file__)) + "/dataset/test/birds_gt.json"
    predictions_file = os.path.dirname(os.path.abspath(__file__)) + '/dataset/test/birds_prediction.json'
    dict_summary = evaluation.evaluate(ground_truth_file, predictions_file, 0.3)
    dict_summary_all = evaluation.accumulate(dict_summary)
    evaluation.summary(dict_summary_all, "./dataset/test/results_report.json")
    