import torch
from models.utils import trim_sequence, trim

	
class AJTWGCNInference(object):
    def __init__(self, args, model, test_loader, label_map, device):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.label_map = label_map

    def predict(self):
        self.model = self.model.eval()
        
        predicted_labels1 = []
        target_labels1 = []
        predicted_labels2 = []
        target_labels2 = []

        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, head_ids, input_mask, segment_ids, dep_ids, level1, level2, level3 = batch
                           
                output1, output2 = self.model(input_ids, head_ids, segment_ids, input_mask, dep_ids)
                
                _,preds1 = torch.max(output1,dim=2)
                _,preds2 = torch.max(output2,dim=2)

                valied_lenght = input_mask.sum(1).tolist()
                final_predict1, target1 = trim(level1, preds1, valied_lenght)
                final_predict2, target2 = trim(level2, preds2, valied_lenght)

                predicted_labels1.extend(final_predict1)  
                target_labels1.extend(target1)   
                predicted_labels2.extend(final_predict2)  
                target_labels2.extend(target2) 
            pred_label1 = trim_sequence(predicted_labels1, target_labels1, self.label_map)
            pred_label2 = trim_sequence(predicted_labels2, target_labels2, self.label_map)

        return [pred_label1, pred_label2]