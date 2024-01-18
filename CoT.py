#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, Conversation, pipeline
import torch


# In[2]:


model_name = "meta-llama/Llama-2-7b-chat-hf"
#name = "RomanOrac/llama-2-7b-slovenian"
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_name = "mistralai/Mistral-7B-Instruct-v0.2"


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[4]:


tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation


# In[5]:


tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate "
            "user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"
            "{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + "
            "message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            )


# In[6]:


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# In[ ]:


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)


# In[8]:


model.generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    #top_k=
)


# In[9]:


generation_pipe = pipeline(
    "conversational",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)


# In[10]:


violence_df = pd.read_csv("violence.csv")


# In[11]:


violence_df.Sentence.loc[0]


# In[12]:


examples_df = violence_df[violence_df["Example"] == True]


# In[13]:


examples_df


# In[14]:


text_df = pd.DataFrame()


# In[15]:


ex_violence = violence_df[violence_df["Example"] == False]
sample_val = ex_violence.sample(5)


# In[16]:


system_prompt = "You are an expert classifier in Slovenian sociology. You always answer in English."


# In[17]:


prompt_base_few_shot = "## Label this Slovenian text, according to wether the whole text is violent, includes hatespeech, calls to violent or subversive actions or sublevation (both explictly or figuratively), or not, indicating 'True' if it does and 'False' if it doesn't:\n"


# In[18]:


FEW_SHOT = prompt_base_few_shot+"# Text: {text}\n# Violent: {label}\n# Description: {description}\n\n"


# In[20]:


prompt_base = "## Label ONLY THE LAST Slovenian text, according to wether the whole text is violent, includes hatespeech, calls to violent or subversive actions or sublevation (both explictly or figuratively), or not, indicating 'True' if it does and 'False' if it doesn't:\n"


# In[21]:


COT = f"""{prompt_base}
## Answer:
Let's think step by step to label this text according to whether the whole text is violent, includes violent methods, suggests planned violent actions or events (both explictly or figuratively), or not.

Step 1: Recognize words or locutions related to violence or calls to violent actions
- Identify key words or locutions related to violence or calls to violent action such as 'žrtva', 'akcija', 'napad', 'konačni obračun', or 'boj'.

Step 2: Recognize words or locutions related to expansionist rhetoric, or words related to foreign conquest, invading or attacking state borders or war.
- Identify key words or locutions related to expansionist rhetoric such as 'mrtvi straži', 'vpasti preko meje', 'or 'osvojiti meje'.

Step 3: Recognize words or locutions related to signs of bodily harm, wounds or mutilation.
- Identify key words or locutions related to signs of bodily harm, such as 'krv', 'rana', or 'smrt'.

Step 4: Recognize words or locutions that show intent or metaphors towards violent actions.
- Identify key words or locutions related to implied violence, such as 'pripraviti', 'zmagovalno', 'otpraviti', or 'izbrisano'.

Step 5: Label the text based on the previous steps
- If you do find instances of violence, calls to violent actions, or national expansionism, label the text as 'Violent: True'
- If you do not find any instances of violence, calls to violent actions, or hatespeech, label the text as 'Violent: False'
- Include a description of why the text has been labeled as such
"""


# In[22]:


text_df["prompts"] = examples_df[["Sentence", "Violent", "Description"]].astype(str).apply(lambda x: FEW_SHOT.format(text=x['Sentence'],label=x['Violent'],description=x['Description']), axis=1)


# In[23]:


few_shots = '## I present you here six (6) examples of correctly labeled texts based on similar problems:\n' + "\n".join(text_df["prompts"].to_list()) + "\n\nUse the previous examples as guidelines without labelling them.\n\n"


# In[24]:


def label(texts):
    conversation = Conversation()
    conversation.add_message({'role':'system', 'content': system_prompt})
    conversation.add_message({'role':'user', 'content': few_shots+COT+prompt_base+"# Text: " + texts + "\n# Let's think step by step: "})
    response = generation_pipe(
        conversation,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
    )
    return response


# In[25]:


sample_val


# In[26]:


label(ex_violence.Sentence[65])


# In[ ]:




