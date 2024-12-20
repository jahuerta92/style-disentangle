import torch
import pytorch_lightning as pl
import transformers
import torch.nn.functional as F
import torch.optim as optim
import copy

from transformers import get_linear_schedule_with_warmup, Adafactor
from torch import nn

## Auxiliary functions
def pooling(embs, mask):
    input_mask_expanded = mask.unsqueeze(-1)
    return torch.sum(embs * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)        

class BaseModelDisentanglement(pl.LightningModule):
    def __init__(self, training_steps=1,
                 eps=1, gamma=1,
                 minibatch_size=8, 
                 warmup_steps=0.1, 
                 lr=2e-5, 
                 dropout=.1, 
                 reset_layers=0,
                 unfrozen_layers=0,
                 with_weights=False,):
        super().__init__()
        self.training_steps = training_steps
        self.eps = eps
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.dropout = dropout
        self.reset_layers = reset_layers
        self.unfrozen_layers = unfrozen_layers
        self.with_weights = with_weights
        
        self.sentence_similarity = transformers.AutoModel.from_pretrained('WhereIsAI/UAE-Large-V1')
        self.sentence_similarity.eval()
        
        self.style_similarity = transformers.AutoModel.from_pretrained('AIDA-UPM/star')
        self.style_pooler = nn.Linear(1024, 1024) # size of embeddings for style and similarity

        self.temperature = torch.nn.Parameter(torch.tensor(0.07))
        self.automatic_optimization = False
        self._freeze_layers()
        
    def _freeze_layers(self):
        # Freeze everything and replace dropout with my own models' dropout
        for param in self.style_similarity.parameters():
            param.requires_grad = False        
        for _, layer in self.style_similarity.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.p = self.dropout
        self.style_similarity.eval()
        
        if self.reset_layers > self.unfrozen_layers:
            self.unfrozen_layers = self.reset_layers
            
        # Unfreeze the last n layers
        if self.unfrozen_layers > 0:
            for layer in self.style_similarity.encoder.layer[-self.unfrozen_layers:]:
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Reset the last n layers entirely
        if self.reset_layers > 0:
            for layer in self.style_similarity.encoder.layer[-self.reset_layers:]:
                for _, l in layer.named_modules():
                    if hasattr(l, 'reset_parameters'):
                        l.reset_parameters()

    def forward(self, x):
        input_ids, attention_mask = x
        style_embedding = self.style_similarity(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        style_embedding = pooling(style_embedding, attention_mask)
        style_embedding = self.style_pooler(style_embedding)
        return style_embedding
    
    def _semantic_forward(self, x):
        input_ids, attention_mask = x
        sentence_embedding = self.sentence_similarity(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        sentence_embedding = pooling(sentence_embedding, attention_mask)
        return sentence_embedding            
    
    def _minibatch(self, batch):
        ids, mask = batch
        id_mb = torch.split(ids, self.minibatch_size)
        mask_mb = torch.split(mask, self.minibatch_size)
        return zip(id_mb, mask_mb)
    
    def _get_similarity(self, x, y, normalize=True):
        x = F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(1), dim=-1) * self.temperature.exp().clamp(-100, 100)
        if normalize:
            return x - torch.max(x, dim=1, keepdim=True)[0].detach()
        else: 
            return x

    def _loss(self, x, y, sem_x, sem_y, labels):
        return 0

    def training_step(self, batch, batch_idx):
        # Unpack batch into texts
        semantic_batch, style_batch = batch
        style_text_1, style_text_2 = style_batch
        semantic_text_1, semantic_text_2 = semantic_batch
        
        # Load the optimizer
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        optimizer.zero_grad()

        loss_tracker, accuracy_tracker = 0, 0
        
        semantic_text_1 = semantic_text_1[0].detach(), semantic_text_1[1].detach()
        semantic_text_2 = semantic_text_2[0].detach(), semantic_text_2[1].detach()
        
        num_chunks = len(style_text_1[0]) // self.minibatch_size
        
        # Compute base embeddings, both style and content
        with torch.no_grad():
            # Style embeddings for minibatching
            reference_1 = torch.cat([self(mb) for mb in self._minibatch(style_text_1)], dim=0)
            reference_2 = torch.cat([self(mb) for mb in self._minibatch(style_text_2)], dim=0)
            
            # Semantic embeddings for penalization
            semantic_embedding_1, semantic_embedding_2 = None, None
            if self.with_weights:
                semantic_embedding_1 = torch.cat([self._semantic_forward(mb) for mb in self._minibatch(semantic_text_1)], dim=0)
                semantic_embedding_2 = torch.cat([self._semantic_forward(mb) for mb in self._minibatch(semantic_text_2)], dim=0)

            # Generate the labels for CCE
            labels = torch.eye(reference_1.shape[0], device=reference_1.device)
            
        # Minibatched one-way InfoNCE for efficient memory usage
        for i, mb in enumerate(self._minibatch(style_text_1)):
            # Copy the references to avoid in-place operations
            copy_reference_1 = copy.deepcopy(reference_1)

            # Compute the style embeddings with gradients
            copy_reference_1[(i*self.minibatch_size):((i+1)*self.minibatch_size)] = self(mb)

            # Compute the style similarity and CCE re-weighted with semantic similarity
            loss, similarity = self._loss(copy_reference_1, reference_2, 
                                          semantic_embedding_1, semantic_embedding_2, 
                                          labels)
            
            with torch.no_grad():
                accuracy = (similarity.argmax(1) == labels.argmax(1)).float().mean()
                loss_tracker+=loss/(2*num_chunks)
                accuracy_tracker+=accuracy/(2*num_chunks)
            
            self.manual_backward(loss)
        
        # Second pass to learn from document set 2 to 1
        for i, mb in enumerate(self._minibatch(style_text_2)):
            # Copy the references to avoid in-place operations
            copy_reference_2 = copy.deepcopy(reference_2)

            # Compute the style embeddings with gradients
            copy_reference_2[(i*self.minibatch_size):((i+1)*self.minibatch_size)] = self(mb)

            # Compute the style similarity and CCE re-weighted with semantic similarity
            loss, similarity = self._loss(copy_reference_2, reference_1, 
                                          semantic_embedding_2, semantic_embedding_1, 
                                          labels)
            
            with torch.no_grad():
                accuracy = (similarity.argmax(1) == labels.argmax(1)).float().mean()
                loss_tracker+=loss/(2*num_chunks)
                accuracy_tracker+=accuracy/(2*num_chunks)
            
            self.manual_backward(loss)

        with torch.no_grad():
            self.log(f'train/loss', loss_tracker)
            self.log(f'train/accuracy', accuracy_tracker)

        optimizer.step()
        lr_scheduler.step()

        return loss_tracker
    
    def validation_step(self, batch, batch_idx, validation=True):
        # Unpack batch into texts
        _, style_batch = batch
        style_text_1, style_text_2 = style_batch
        
        style_embs_1 = self(style_text_1)
        style_embs_2 = self(style_text_2)
        
        similarity = F.cosine_similarity(style_embs_1.unsqueeze(0), style_embs_2.unsqueeze(1), dim=-1)
        labels = torch.eye(similarity.shape[0], device=similarity.device)
        cat_labels = labels.argmax(-1)
        loss_1 = F.cross_entropy(similarity, cat_labels)
        loss_2 = F.cross_entropy(similarity.T, cat_labels)
        loss = loss_1 + loss_2
        
        accuracy = (similarity.argmax(1) == cat_labels).float().mean()

        txt = 'val' if validation else 'test'
        self.log(f'{txt}/loss', loss)
        self.log(f'{txt}/accuracy', accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, validation=False)
    
    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), 
                              scale_parameter=False,
                              relative_step=False,
                              warmup_init=False,
                              lr=self.lr,
                              weight_decay=1e-4,
                              )
        
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                    num_warmup_steps=self.training_steps*self.warmup_steps, 
                                                    num_training_steps=self.training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": 'step',
                "strict": True,
                "name": 'linear_schedule_with_warmup',
            },
    }

class SimilarityWeightLossModel(BaseModelDisentanglement):
    def _weighted_ce(self, logits, labels, weights):
        with torch.no_grad():
            ce_weight = F.cross_entropy(weights, labels, reduction='none')
            ce_weight = torch.exp(ce_weight)
            ce_weight = (ce_weight / ce_weight.mean()) 
        
        cc_loss = F.cross_entropy(logits, labels, reduction='none') 

        return torch.mean(cc_loss * ce_weight) 

    def _loss(self, x, y, sem_x, sem_y, labels):
        similarity = self._get_similarity(x, y)
        if sem_x is None or sem_y is None:
            loss = F.cross_entropy(similarity, labels.argmax(-1)) + F.cross_entropy(similarity.T, labels.argmax(-1))
            return loss, similarity
        else:
            sem_similarity = self._get_similarity(sem_x, sem_y)
            loss_1 = self._weighted_ce(x, similarity, labels, sem_similarity)
            loss_2 = self._weighted_ce(y, similarity.T, labels, sem_similarity.T)
            return loss_1+loss_2, similarity

class SyntheticHardNegativesLossModel(BaseModelDisentanglement):
    def _infonce_with_fakes(self, x, self_sim, other_sim, similarity, labels):
        with_self = self._get_similarity(x, self_sim, normalize=False)
        with_self *= 1 - torch.eye(with_self.shape[0], device=with_self.device) # remove self similarity op, set to 0
        with_others = self._get_similarity(x, other_sim, normalize=False)
 
        expanded_labels = torch.cat([labels, torch.zeros_like(labels), torch.zeros_like(labels)], dim=1)
        expanded_logits = torch.cat([similarity, with_self, with_others], dim=1)
        expanded_logits = expanded_logits - torch.max(expanded_logits, dim=1, keepdim=True)[0].detach()
        
        loss = F.cross_entropy(expanded_logits, expanded_labels.argmax(1))
        return loss
    
    def _loss(self, x, y, sim_x, sim_y, labels):
        if sim_x is None or sim_y is None:
            similarity = self._get_similarity(x, y)
            loss = F.cross_entropy(similarity, labels.argmax(1))  + F.cross_entropy(similarity.T, labels.argmax(1))
            return loss, similarity
        else:
            similarity = self._get_similarity(x, y, normalize=False)
            loss = self._infonce_with_fakes(x, sim_x, sim_y, similarity, labels) + self._infonce_with_fakes(y, sim_y, sim_x, similarity.T, labels)
            return loss, similarity

#This is the good one!
class SyntheticHardNegativesLossNonZeroSelfModel(SyntheticHardNegativesLossModel):
    def _infonce_with_fakes(self, x, self_sim, other_sim, similarity, labels):
        with_self = self._get_similarity(x, self_sim, normalize=False)
        with_others = self._get_similarity(x, other_sim, normalize=False)
 
        expanded_labels = torch.cat([labels, torch.zeros_like(labels), torch.zeros_like(labels)], dim=1)
        expanded_logits = torch.cat([similarity, with_self, with_others], dim=1)
        expanded_logits = expanded_logits - torch.max(expanded_logits, dim=1, keepdim=True)[0].detach()
        
        loss = F.cross_entropy(expanded_logits, expanded_labels.argmax(1))
        return loss
    
class SyntheticHardNegativesWithSelfLossModel(SyntheticHardNegativesLossModel):
    def _infonce_with_fakes(self, x, self_sim, other_sim, similarity, labels):
        with_self = self._get_similarity(x, self_sim, normalize=False)
        with_others = self._get_similarity(x, other_sim, normalize=False)
        extra_style_negatives = self._get_similarity(x, x, normalize=False)
        extra_style_negatives *= 1 - torch.eye(extra_style_negatives.shape[0], device=extra_style_negatives.device) # Self-similarity is 1, remove it
        
        expanded_labels = torch.cat([labels, torch.zeros_like(labels), torch.zeros_like(labels), torch.zeros_like(labels)], dim=1)
        expanded_logits = torch.cat([similarity, with_self, with_others, extra_style_negatives], dim=1)
        expanded_logits = expanded_logits - torch.max(expanded_logits, dim=1, keepdim=True)[0].detach()
        
        loss = F.cross_entropy(expanded_logits, expanded_labels.argmax(1))
        return loss

## DEPRECATED MODELS
class AuthorModel(pl.LightningModule):
    def __init__(self, training_steps=1,
                 eps=1, gamma=1,
                 minibatch_size=8, 
                 warmup_steps=0.1, 
                 lr=2e-5, 
                 dropout=.1, 
                 reset_layers=0,
                 unfrozen_layers=0,
                 with_weights=False,):
        super().__init__()
        self.training_steps = training_steps
        self.eps = eps
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.dropout = dropout
        self.reset_layers = reset_layers
        self.unfrozen_layers = unfrozen_layers
        self.with_weights = with_weights
        
        self.sentence_similarity = transformers.AutoModel.from_pretrained('WhereIsAI/UAE-Large-V1')
        self.sentence_similarity.eval()
        
        self.style_similarity = transformers.AutoModel.from_pretrained('AIDA-UPM/star')
        self.style_pooler = nn.Linear(1024, 1024)
        self.temperature = torch.nn.Parameter(torch.tensor(0.07))
        self.automatic_optimization = False
        self._freeze_layers()
        
    def _freeze_layers(self):
        # Freeze everything and replace dropout with my own models' dropout
        for param in self.style_similarity.parameters():
            param.requires_grad = False        
        for _, layer in self.style_similarity.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.p = self.dropout
        self.style_similarity.eval()
        
        if self.reset_layers > self.unfrozen_layers:
            self.unfrozen_layers = self.reset_layers
            
        # Unfreeze the last n layers
        if self.unfrozen_layers > 0:
            for layer in self.style_similarity.encoder.layer[-self.unfrozen_layers:]:
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Reset the last n layers entirely
        if self.reset_layers > 0:
            for layer in self.style_similarity.encoder.layer[-self.reset_layers:]:
                for _, l in layer.named_modules():
                    if hasattr(l, 'reset_parameters'):
                        l.reset_parameters()

    def forward(self, x):
        input_ids, attention_mask = x
        style_embedding = self.style_similarity(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        style_embedding = pooling(style_embedding, attention_mask)
        style_embedding = self.style_pooler(style_embedding)
        return style_embedding
    
    def _semantic_forward(self, x):
        input_ids, attention_mask = x
        sentence_embedding = self.sentence_similarity(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        sentence_embedding = pooling(sentence_embedding, attention_mask)
        return sentence_embedding            
    
    def _minibatch(self, batch):
        ids, mask = batch
        id_mb = torch.split(ids, self.minibatch_size)
        mask_mb = torch.split(mask, self.minibatch_size)
        return zip(id_mb, mask_mb)        
    
    def _weighted_ce(self, logits, labels, weights):
        '''
            # Increase loss when the problem is hard to predict for similarity, penalize failures more in these edge cases
            instance_difficulty_weight = F.cross_entropy(weights, labels, reduction='none') + self.eps 
            # Increase loss when the logits fail to choose the positive in presence of a hard negative
            # If the model chooses an easy positive, the loss is not increased, thus we invert the diagonal.
            difficulty_per_pair = torch.where(labels == 1, torch.exp(1-weights), weights)
            class_difficulty_weight = torch.gather(difficulty_per_pair, 1, 
                                                   logits.argmax(1).unsqueeze(1)).squeeze() + self.gamma
         '''
        with torch.no_grad():
            '''
            # Increase loss when the problem is hard to predict for similarity, penalize failures more in these edge cases
            instance_difficulty_weight = F.cross_entropy(weights, labels, reduction='none') + self.eps 
            '''
            ce_weight = F.cross_entropy(weights, labels, reduction='none')
            ce_weight = torch.exp(ce_weight)
            ce_weight = (ce_weight / ce_weight.mean()) 
        
        cc_loss = F.cross_entropy(logits, labels, reduction='none') 

        return torch.mean(cc_loss * ce_weight) 

    def _weighted_infonce(self, x, y, labels, weights=None):
        similarity = F.cosine_similarity(x.unsqueeze(0), 
                                         y.unsqueeze(1),
                                         dim=-1) * self.temperature.exp().clamp(-100, 100)
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()
        
        if weights is None:
            loss_1 = F.cross_entropy(logits, labels.argmax(1))
            loss_2 = F.cross_entropy(logits.T, labels.argmax(1))
            loss = loss_1 + loss_2
            return loss, similarity
        
        else:    
            loss_1 = self._weighted_ce(logits, labels, weights)
            loss_2 = self._weighted_ce(logits.T, labels, weights.T)
            loss = loss_1 + loss_2 
            return loss, similarity
        
    def training_step(self, batch, batch_idx):
        # Unpack batch into texts
        semantic_batch, style_batch = batch
        style_text_1, style_text_2 = style_batch
        semantic_text_1, semantic_text_2 = semantic_batch
        
        # Load the optimizer
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        optimizer.zero_grad()

        loss_tracker, accuracy_tracker = [], []
        
        semantic_text_1 = semantic_text_1[0].detach(), semantic_text_1[1].detach()
        semantic_text_2 = semantic_text_2[0].detach(), semantic_text_2[1].detach()
        style_text_2 = style_text_2[0].detach(), style_text_2[1].detach()
        
        num_chunks = len(style_text_1[0]) // self.minibatch_size
        
        # Compute base embeddings, both style and content
        with torch.no_grad():
            # Style embeddings for minibatching
            reference_1 = torch.cat([self(mb) for mb in self._minibatch(style_text_1)], dim=0)
            reference_2 = torch.cat([self(mb) for mb in self._minibatch(style_text_2)], dim=0)
            
            # Semantic embeddings for penalization
            semantic_similarity = None
            if self.with_weights:
                semantic_embedding_1 = torch.cat([self._semantic_forward(mb) for mb in self._minibatch(semantic_text_1)], dim=0)
                semantic_embedding_2 = torch.cat([self._semantic_forward(mb) for mb in self._minibatch(semantic_text_2)], dim=0)
                semantic_similarity = F.cosine_similarity(semantic_embedding_1.unsqueeze(0), 
                                                        semantic_embedding_2.unsqueeze(1), dim=-1)
                
            # Generate the labels for CCE
            labels = torch.eye(reference_1.shape[0], device=reference_1.device)
            
        # Minibatched one-way InfoNCE for efficient memory usage
        for i, mb in enumerate(self._minibatch(style_text_1)):
            # Copy the references to avoid in-place operations
            copy_reference = copy.deepcopy(reference_1)
            
            # Compute the style embeddings with gradients
            copy_reference[(i*self.minibatch_size):((i+1)*self.minibatch_size)] = self(mb)
            
            # Compute the style similarity and CCE re-weighted with semantic similarity
            loss, similarity = self._weighted_infonce(copy_reference, reference_2, labels, semantic_similarity)
            
            with torch.no_grad():
                accuracy = (similarity.argmax(1) == labels.argmax(1)).float().mean()
                loss_tracker.append(loss)
                accuracy_tracker.append(accuracy)
            
            self.manual_backward(loss)

        with torch.no_grad():
            final_loss = sum(loss_tracker)/num_chunks
            self.log(f'train/loss', final_loss)
            self.log(f'train/accuracy', sum(accuracy_tracker)/num_chunks)
        
        optimizer.step()
        lr_scheduler.step()

        return final_loss
    
    def validation_step(self, batch, batch_idx, validation=True):
        # Unpack batch into texts
        _, style_batch = batch
        style_text_1, style_text_2 = style_batch
        
        style_embs_1 = self(style_text_1)
        style_embs_2 = self(style_text_2)
        
        similarity = F.cosine_similarity(style_embs_1.unsqueeze(0), style_embs_2.unsqueeze(1), dim=-1)
        labels = torch.eye(similarity.shape[0], device=similarity.device)
        cat_labels = labels.argmax(-1)
        loss_1 = F.cross_entropy(similarity, cat_labels)
        loss_2 = F.cross_entropy(similarity.T, cat_labels)
        loss = loss_1 + loss_2
        
        accuracy = (similarity.argmax(1) == cat_labels).float().mean()

        txt = 'val' if validation else 'test'
        self.log(f'{txt}/loss', loss)
        self.log(f'{txt}/accuracy', accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, validation=False)
    
    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), 
                              scale_parameter=False,
                              relative_step=False,
                              warmup_init=False,
                              lr=self.lr,
                              weight_decay=1e-4,
                              )
        
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                    num_warmup_steps=self.training_steps*self.warmup_steps, 
                                                    num_training_steps=self.training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": 'step',
                "frequency": 1,
                "strict": True,
                "name": 'linear_schedule_with_warmup',
            },
    }

class AuthorModelDisentanglement(pl.LightningModule):
    def __init__(self, training_steps=1,
                 eps=1, gamma=1,
                 minibatch_size=8, 
                 warmup_steps=0.1, 
                 lr=2e-5, 
                 dropout=.1, 
                 reset_layers=0,
                 unfrozen_layers=0,
                 with_weights=False,):
        super().__init__()
        self.training_steps = training_steps
        self.eps = eps
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.dropout = dropout
        self.reset_layers = reset_layers
        self.unfrozen_layers = unfrozen_layers
        self.with_weights = with_weights
        
        self.sentence_similarity = transformers.AutoModel.from_pretrained('WhereIsAI/UAE-Large-V1')
        self.sentence_similarity.eval()
        
        self.style_similarity = transformers.AutoModel.from_pretrained('AIDA-UPM/star')
        self.style_pooler = nn.Linear(1024, 1024) # size of embeddings for style and similarity

        self.temperature = torch.nn.Parameter(torch.tensor(0.07))
        self.automatic_optimization = False
        self._freeze_layers()
        
    def _freeze_layers(self):
        # Freeze everything and replace dropout with my own models' dropout
        for param in self.style_similarity.parameters():
            param.requires_grad = False        
        for _, layer in self.style_similarity.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.p = self.dropout
        self.style_similarity.eval()
        
        if self.reset_layers > self.unfrozen_layers:
            self.unfrozen_layers = self.reset_layers
            
        # Unfreeze the last n layers
        if self.unfrozen_layers > 0:
            for layer in self.style_similarity.encoder.layer[-self.unfrozen_layers:]:
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Reset the last n layers entirely
        if self.reset_layers > 0:
            for layer in self.style_similarity.encoder.layer[-self.reset_layers:]:
                for _, l in layer.named_modules():
                    if hasattr(l, 'reset_parameters'):
                        l.reset_parameters()

    def forward(self, x):
        input_ids, attention_mask = x
        style_embedding = self.style_similarity(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        style_embedding = pooling(style_embedding, attention_mask)
        style_embedding = self.style_pooler(style_embedding)
        return style_embedding
    
    def _semantic_forward(self, x):
        input_ids, attention_mask = x
        sentence_embedding = self.sentence_similarity(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        sentence_embedding = pooling(sentence_embedding, attention_mask)
        return sentence_embedding            
    
    def _minibatch(self, batch):
        ids, mask = batch
        id_mb = torch.split(ids, self.minibatch_size)
        mask_mb = torch.split(mask, self.minibatch_size)
        return zip(id_mb, mask_mb)
    
    def _get_similarity(self, x, y):
        x = F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(1), dim=-1) * self.temperature.exp().clamp(-100, 100)
        return x - torch.max(x, dim=1, keepdim=True)[0].detach()
    
    def _infonce_with_fakes(self, x, self_sim, other_sim, similarity, labels):
        with_self = F.cosine_similarity(x.unsqueeze(0),
                                        self_sim.unsqueeze(1),
                                        dim=-1) * self.temperature.exp().clamp(-100, 100)
        with_others = F.cosine_similarity(x.unsqueeze(0),
                                          other_sim.unsqueeze(1),
                                          dim=-1) * self.temperature.exp().clamp(-100, 100)

        expanded_labels = torch.cat([labels, torch.zeros_like(labels), torch.zeros_like(labels)], dim=1)
        
        loss = F.cross_entropy(torch.cat([similarity, with_self, with_others], dim=1), expanded_labels.argmax(1))
        return loss
    
    def _loss(self, x, y, sim_x, sim_y, labels):
        similarity = self._get_similarity(x, y)
        if sim_x is None or sim_y is None:
            loss = F.cross_entropy(similarity, labels.argmax(1))
            return loss, similarity
        else:
            loss_1 = self._infonce_with_fakes(x, sim_x, sim_y, similarity, labels)
            loss_2 = self._infonce_with_fakes(y, sim_y, sim_x, similarity.T, labels)
            loss = loss_1 + loss_2
            return loss, similarity

    def training_step(self, batch, batch_idx):
        # Unpack batch into texts
        semantic_batch, style_batch = batch
        style_text_1, style_text_2 = style_batch
        semantic_text_1, semantic_text_2 = semantic_batch
        
        # Load the optimizer
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        optimizer.zero_grad()

        loss_tracker, accuracy_tracker = 0, 0
        
        semantic_text_1 = semantic_text_1[0].detach(), semantic_text_1[1].detach()
        semantic_text_2 = semantic_text_2[0].detach(), semantic_text_2[1].detach()
        
        num_chunks = len(style_text_1[0]) // self.minibatch_size
        
        # Compute base embeddings, both style and content
        with torch.no_grad():
            # Style embeddings for minibatching
            reference_1 = torch.cat([self(mb) for mb in self._minibatch(style_text_1)], dim=0)
            reference_2 = torch.cat([self(mb) for mb in self._minibatch(style_text_2)], dim=0)
            
            # Semantic embeddings for penalization
            semantic_embedding_1, semantic_embedding_2 = None, None
            if self.with_weights:
                semantic_embedding_1 = torch.cat([self._semantic_forward(mb) for mb in self._minibatch(semantic_text_1)], dim=0)
                semantic_embedding_2 = torch.cat([self._semantic_forward(mb) for mb in self._minibatch(semantic_text_2)], dim=0)

            # Generate the labels for CCE
            labels = torch.eye(reference_1.shape[0], device=reference_1.device)
            
        # Minibatched one-way InfoNCE for efficient memory usage
        for i, mb_1 in enumerate(self._minibatch(style_text_1)):
            # Copy the references to avoid in-place operations
            copy_reference_1 = copy.deepcopy(reference_1)

            # Compute the style embeddings with gradients
            copy_reference_1[(i*self.minibatch_size):((i+1)*self.minibatch_size)] = self(mb_1)

            # Compute the style similarity and CCE re-weighted with semantic similarity
            loss, similarity = self._loss(copy_reference_1, reference_2, 
                                          semantic_embedding_1, semantic_embedding_2, 
                                          labels)
            
            with torch.no_grad():
                accuracy = (similarity.argmax(1) == labels.argmax(1)).float().mean()
                loss_tracker+=loss/num_chunks
                accuracy_tracker+=accuracy/num_chunks
            
            self.manual_backward(loss)

        with torch.no_grad():
            self.log(f'train/loss', loss_tracker)
            self.log(f'train/accuracy', accuracy_tracker)

        optimizer.step()
        lr_scheduler.step()

        return loss_tracker
    
    def validation_step(self, batch, batch_idx, validation=True):
        # Unpack batch into texts
        _, style_batch = batch
        style_text_1, style_text_2 = style_batch
        
        style_embs_1 = self(style_text_1)
        style_embs_2 = self(style_text_2)
        
        similarity = F.cosine_similarity(style_embs_1.unsqueeze(0), style_embs_2.unsqueeze(1), dim=-1)
        labels = torch.eye(similarity.shape[0], device=similarity.device)
        cat_labels = labels.argmax(-1)
        loss_1 = F.cross_entropy(similarity, cat_labels)
        loss_2 = F.cross_entropy(similarity.T, cat_labels)
        loss = loss_1 + loss_2
        
        accuracy = (similarity.argmax(1) == cat_labels).float().mean()

        txt = 'val' if validation else 'test'
        self.log(f'{txt}/loss', loss)
        self.log(f'{txt}/accuracy', accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, validation=False)
    
    def configure_optimizers(self):
        optimizer = Adafactor(self.parameters(), 
                              scale_parameter=False,
                              relative_step=False,
                              warmup_init=False,
                              lr=self.lr,
                              weight_decay=1e-4,
                              )
        
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                    num_warmup_steps=self.training_steps*self.warmup_steps, 
                                                    num_training_steps=self.training_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
                "interval": 'step',
                "strict": True,
                "name": 'linear_schedule_with_warmup',
            },
    }