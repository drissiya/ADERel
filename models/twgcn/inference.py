import torch
from models.utils import trim_sequence, trim

	
class TWGCNInference(object):
    def __init__(self, args, model, test_loader, label_map, device):
        self.args = args
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.label_map = label_map

    def predict(self):
        self.model = self.model.eval()
        
        predicted_labels = []
        target_labels = []

        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, head_ids, input_mask, segment_ids, dep_ids, level1 = batch
                           
                output = self.model(input_ids, head_ids, segment_ids, input_mask, dep_ids)
                valied_lenght = input_mask.sum(1).tolist()   
              
                _,preds = torch.max(output,dim=2)               

                final_predict, target = trim(level1, preds, valied_lenght)

                predicted_labels.extend(final_predict)  
                target_labels.extend(target)   

            pred_labels = trim_sequence(predicted_labels, target_labels, self.label_map)

        return pred_labels