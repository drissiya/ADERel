import torch
import numpy as np

class TWGCNLearner(object):
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
        correct_predictions = 0
        for batch in self.train_loader:
            batch = tuple(t.to(self.device) for t in batch)
            self.optimizer.zero_grad()

            input_ids, head_ids, input_mask, segment_ids, dep_ids, level = batch
            output = self.model(input_ids, head_ids, segment_ids, input_mask, dep_ids)   
                
            _,preds = torch.max(output,dim=2)
            output = output.view(-1,output.shape[-1])  

            b_labels_shaped = level.view(-1)

            loss = self.loss_fn(output,b_labels_shaped)
            correct_predictions += torch.sum(preds == level)
            losses.append(loss.item())

            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
        return correct_predictions.double()/len(self.train_loader), np.mean(losses)

    def train(self):
        best_valid_loss = float('inf')
        normalizer = self.args.batch_size*self.args.max_seq_length
        for epoch in range(self.args.epochs): 
            print(f'Epoch: {epoch+1:02}')          
            train_acc, train_loss = self.train_epoch()
            train_acc = train_acc/normalizer
                        
            val_acc, val_loss = self.evaluate()
            val_acc = val_acc/normalizer


            self.scheduler.step()

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                model_save = F"{self.args.dataset_name}_model.bin"
                path = F"{self.args.output_dir}/{model_save}" 
                torch.save(self.model.state_dict(), path)

            print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc:.2f}')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Accuracy: {val_acc:.2f}')
            
    def evaluate(self):
        self.model = self.model.eval()       
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for step, batch in enumerate(self.valid_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, head_ids, input_mask, segment_ids, dep_ids, level = batch
                           
                output = self.model(input_ids, head_ids, segment_ids, input_mask, dep_ids)
                
                _,preds = torch.max(output,dim=2)
                output = output.view(-1,output.shape[-1])

                b_labels_shaped = level.view(-1)

                loss = self.loss_fn(output,b_labels_shaped)
                correct_predictions += torch.sum(preds == level)

                losses.append(loss.item())
                
        return correct_predictions.double()/len(self.valid_loader), np.mean(losses)