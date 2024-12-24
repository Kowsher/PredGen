from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from typing import List, Optional, Union
from transformers.cache_utils import Cache
import torch.nn.functional as F
import random

class PredictorCausalLM(PreTrainedModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.llm = AutoModelForCausalLM.from_pretrained(config._name_or_path)  # Load the model with config
        #self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)  # Load the model with config
        self.gen_tok = config.nub_of_token_generation
        self.pad_token_id = config.pad_token_id
        
        if int(num_labels) == 1:
            self.problem_type = "regression"
            raw_weights = torch.linspace(self.gen_tok, 2, steps=self.gen_tok)
            self.normalized_weights = raw_weights * (len(raw_weights) / raw_weights.sum())
            self.loss_fct = CrossEntropyLoss(reduction='none')
        else:
            self.problem_type = "classification"
            self.dense_head = nn.Linear(config.vocab_size, config.dense_representation, bias=False)
            self.score = nn.Linear(config.dense_representation, self.num_labels, bias=False)
            self.loss_fct = CrossEntropyLoss()
            

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        output_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_inference: Optional[bool] = False,  # Add flag to control behavior during inference
        sampling_prob: float = 0.0,  # Sampling probability for scheduled sampling
        **kwargs,
    ):
        #print(output_ids[0], self.tokenizer.decode(output_ids[0]))

        if self.training:
            # Training mode: use the entire sequence (including the last n tokens)
          
            input_ids = torch.cat((input_ids, output_ids), dim=1)
            attention_mask = torch.cat((attention_mask, output_attention_mask), dim=1)
            gen_tok = output_ids.shape[1]
    
            llm_labels = input_ids.clone()
            llm_labels[llm_labels == self.pad_token_id] = -100    
            #llm_labels[:, :-gen_tok] = -100
        
            truncated_input_ids = input_ids[:, :-gen_tok]  # Remove last n tokens for generation

            truncated_attention_mask = attention_mask[:, :-gen_tok]

            #print('llm_labels', llm_labels[0])


            with torch.no_grad():

                for _ in range(gen_tok):
                    outputs = self.llm(
                        input_ids=truncated_input_ids,
                        attention_mask=truncated_attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True,
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    
                    # Decide whether to use the ground truth or the generated token
                    use_generated_token = random.random() < sampling_prob
                    next_input_token = next_token_id if use_generated_token else input_ids[:, truncated_input_ids.size(1)].unsqueeze(1)
                    next_attention_mask = (next_token_id != self.pad_token_id) if use_generated_token else attention_mask[:, truncated_input_ids.size(1)].unsqueeze(1)

                    truncated_input_ids = torch.cat([truncated_input_ids, next_input_token], dim=1)
                    truncated_attention_mask = torch.cat([truncated_attention_mask, next_attention_mask], dim=1)

            # Get final output logits
            outputs = self.llm(
                input_ids=truncated_input_ids,
                attention_mask=truncated_attention_mask,
                labels=llm_labels, 
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, # Pass LLM labels for language modeling loss
                return_dict=True,
            )
  
            hidden_states = outputs.logits  # Logits from the model
            llm_loss = outputs.loss  # Loss from the language model (if labels are provided)            

        # Inference mode
        else:
            if output_ids is not None:
                gen_tok = output_ids.shape[1]
                llm_labels = torch.cat((input_ids, output_ids), dim=1)
                #llm_labels[:, :-gen_tok] = -100
                llm_labels[llm_labels == self.pad_token_id] = -100 
            else:
                gen_tok = self.gen_tok
                llm_labels = None
            
            truncated_input_ids = input_ids.clone()  
            truncated_attention_mask = attention_mask.clone()

            # Generate tokens one by one for the next n steps
            
            for _ in range(gen_tok):
                outputs = self.llm(
                    input_ids=truncated_input_ids,
                    attention_mask=truncated_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=True,  # Use cache to optimize generation
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                )
                
                # Get the generated token logits and take argmax to simulate greedy decoding
                next_token_logits = outputs.logits[:, -1, :]  # Get logits for the last token
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # Get the predicted token
                
                # Append the generated token to the input for the next iteration
                truncated_input_ids = torch.cat([truncated_input_ids, next_token_id], dim=1)

                non_pad_mask = (next_token_id != self.pad_token_id).long()
                truncated_attention_mask = torch.cat([truncated_attention_mask, non_pad_mask], dim=1)
            
            # Now `truncated_input_ids` contains the generated tokens as well
            # Get the hidden states from the LLM for the last n tokens
            outputs = self.llm(
                input_ids=truncated_input_ids,
                attention_mask=truncated_attention_mask,
                labels=llm_labels,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            
            hidden_states = outputs.logits  # Hidden states for the last n tokens
            llm_loss = outputs.loss

    
        # Initialize loss variable
        loss = None

        if self.problem_type == 'regression':
            hidden_states = hidden_states[:, -gen_tok:, :]
            pooled_logits = hidden_states.clone()
            if output_ids is not None and labels is not None:
                labels = llm_labels[:, -gen_tok:]
                truth_tokens_flat = labels.reshape(-1)
                
                
                token_losses = self.loss_fct(pooled_logits.reshape(-1, hidden_states.size(-1)) , truth_tokens_flat)  # Shape: (batch_size * seq_len)
                token_losses = token_losses.reshape(labels.size())   # Reshape to (batch_size, seq_len)
                #print(token_losses.shape, self.normalized_weights[:gen_tok].unsqueeze(0).shape, gen_tok)
                weighted_losses = token_losses * self.normalized_weights[:gen_tok].unsqueeze(0).to(token_losses.device)  # Broadcast weights to (batch_size, seq_len)
                loss = weighted_losses.mean() 
        else:
            hidden_states = hidden_states[:, -self.gen_tok:, :]
            hidden_states = self.dense_head(hidden_states)
            pooled_logits = self.score(hidden_states.mean(dim=1))  # Mean pooling over the last n tokens
            if output_ids is not None and labels is not None:
                if labels.dtype == torch.long or labels.dtype == torch.int:
                    loss = self.loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
                
                else:
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(pooled_logits, labels)
    
        if loss is not None and llm_loss is not None:
            if llm_loss is not None:

                log_b = torch.log(1+llm_loss)
                log_c = torch.log(1+loss)
                #print(llm_loss, loss)
                    
         
                mx = torch.max(loss, llm_loss) + 1
                loss = mx  + mx * torch.exp(-torch.abs(log_b - log_c))
                #loss = llm_loss + loss  + mx * torch.exp(-torch.abs(log_b - log_c))


    
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
        )

