import torch
import numpy as np

class AJTWGCNLearner(object):
    def __init__(self, args, model, optimizer, train_loader, valid_loader, loss_fn, device, scheduler):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self.device = device

    def train_epoch(self):
        self.model = self.model.train()
        losses = []
        correct_predictions1 = 0
        correct_predictions2 = 0
        for batch in self.train_loader:
            batch = tuple(t.to(self.device) for t in batch)
            self.optimizer.zero_grad()

            input_ids, head_ids, input_mask, segment_ids, dep_ids, level1, level2, level3 = batch
            output1, output2 = self.model(input_ids, head_ids, segment_ids, input_mask, dep_ids)   
                
            _,preds1 = torch.max(output1,dim=2)
            _,preds2 = torch.max(output2,dim=2)

            output1 = output1.view(-1,output1.shape[-1])
            output2 = output2.view(-1,output2.shape[-1])

            b_labels_shaped1 = level1.view(-1)
            b_labels_shaped2 = level2.view(-1)

            outputs = [output1, output2]
            b_labels_shaped = [b_labels_shaped1, b_labels_shaped2]

            loss = self.loss_fn(outputs,b_labels_shaped)
            correct_predictions1 += torch.sum(preds1 == level1)
            correct_predictions2 += torch.sum(preds2 == level2)
            losses.append(loss.item())

            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        return correct_predictions1.double()/len(self.train_loader), correct_predictions2.double()/len(self.train_loader), np.mean(losses)

    def train(self):
        best_valid_loss = float('inf')
        normalizer = self.args.batch_size*self.args.max_seq_length
        for epoch in range(self.args.epochs): 
            print(f'Epoch: {epoch+1:02}')          
            train_acc_l1, train_acc_l2, train_loss = self.train_epoch()
            train_acc_l1 = train_acc_l1/normalizer
            train_acc_l2 = train_acc_l2/normalizer
                        
            val_acc_l1, val_acc_l2, val_loss = self.evaluate()
            val_acc_l1 = val_acc_l1/normalizer
            val_acc_l2 = val_acc_l2/normalizer

            self.scheduler.step()

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                model_save = F"{self.args.dataset_name}_model.bin"
                path = F"{self.args.output_dir}/{model_save}" 
                torch.save(self.model.state_dict(), path)

            print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy level 1: {train_acc_l1:.2f} | Train Accuracy level 2: {train_acc_l2:.2f}')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Accuracy level 1: {val_acc_l1:.2f} |  Val. Accuracy level 2: {val_acc_l2:.2f}')

    def evaluate(self):
        self.model = self.model.eval()       
        losses = []
        correct_predictions1 = 0
        correct_predictions2 = 0
        with torch.no_grad():
            for step, batch in enumerate(self.valid_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, head_ids, input_mask, segment_ids, dep_ids, level1, level2, level3 = batch
                           
                output1, output2 = self.model(input_ids, head_ids, segment_ids, input_mask, dep_ids)
                
                _,preds1 = torch.max(output1,dim=2)
                _,preds2 = torch.max(output2,dim=2)

                output1 = output1.view(-1,output1.shape[-1])
                output2 = output2.view(-1,output2.shape[-1])

                b_labels_shaped1 = level1.view(-1)
                b_labels_shaped2 = level2.view(-1)

                outputs = [output1, output2]
                b_labels_shaped = [b_labels_shaped1, b_labels_shaped2]

                #correct_predictions += torch.sum(preds1 == level1)
                loss = self.loss_fn(outputs,b_labels_shaped)
                correct_predictions1 += torch.sum(preds1 == level1)
                correct_predictions2 += torch.sum(preds2 == level2)
                losses.append(loss.item())
                
        return correct_predictions1.double()/len(self.valid_loader), correct_predictions2.double()/len(self.valid_loader), np.mean(losses)