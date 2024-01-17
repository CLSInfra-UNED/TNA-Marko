# Text Classification Assistant

This Python script serves as an assistant for classifying Slovenian text regarding its content, specifically determining if it contains elements of violence, hate speech, or calls to violent or subversive actions or subversion. It leverages the `transformers` library, utilizing a pre-trained language model.

## Usage

1. Install the required libraries:

   ```bash
   pip install pandas torch transformers
   ```

2. Download the pre-trained language model:

   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

   model_name = "mistralai/Mistral-7B-Instruct-v0.2"
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```

3. Set up the language model:

   ```python
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.float16,
       bnb_4bit_use_double_quant=True,
   )

   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       quantization_config=bnb_config,
       device_map="auto",
       trust_remote_code=True,
   )

   model.generation_config = GenerationConfig(
       do_sample=True,
       temperature=0.7,
   )
   ```

4. Read the input data:

   ```python
   import pandas as pd

   violence_df = pd.read_csv("violence.csv")
   ```

5. Define the classification prompts:

   ```python
   system_prompt = "You are an expert classifier in Slovenian sociology. You always answer in English."

   prompt_base_few_shot = "## Label this Slovenian text, according to whether the whole text is violent, includes hate speech, calls to violent or subversive actions or subversion (both explicitly or figuratively), or not, indicating 'True' if it does and 'False' if it doesn't:\n"

   FEW_SHOT = prompt_base_few_shot + "# Text: {text}\n# Violent: {label}\n# Description: {description}\n\n"
   ```

6. Classify text samples:

   ```python
   def label(texts):
       # ... (rest of the code)
   ```

   Replace `texts` with the text you want to classify.


## Chain of Thought Examples and Explanation

To guide the text classification process, the script employs a chain of thought (COT) methodology. This section provides a detailed explanation of the chain of thought and includes examples to illustrate the process.

### Chain of Thought (COT) Methodology

The COT methodology is a systematic approach used to classify Slovenian text based on the presence or absence of elements such as violence, hate speech, or calls to violent actions. The process is divided into steps, helping the classifier break down the analysis into manageable components.

#### Step 1: Recognize Words or Locutions Related to Violence or Calls to Violent Actions

- Identify key words or expressions associated with violence or calls to violent action, such as 'bitka' (battle), 'upor' (rebellion), 'strmoglavljenje vlade' (government overthrow), or 'boj' (fight).

#### Step 2: Recognize Words or Locutions Related to Hate Speech

- Identify key words or expressions associated with hate speech, such as 'strupeni Žid' (poisoned Jew), 'tujci ven' (foreigners out), 'Čefur' (derogatory term for certain ethnic groups), or 'kurbe' (whores).

#### Step 3: Label the Text Based on the Previous Steps

- If instances of violence, calls to violent actions, or hate speech are identified, label the text as 'Violent: True.'
- If no instances are found, label the text as 'Violent: False.'
- Provide a description of why the text received the specified label.

### Examples

The script includes a set of few-shot examples to assist the classifier in understanding the labeling process. Each example consists of a Slovenian text, its corresponding label (True or False), and a description explaining the classification rationale.

#### Few-Shot Examples

The few-shot examples serve as guidelines for classifying similar texts. They are presented in a format that includes the text, the expected label, and a description. The classifier is encouraged to use these examples as reference points when labeling new texts.

### Usage

1. The classifier is prompted with a few-shot example, providing context and a template for classification.
2. The provided examples are intended to assist in understanding the thought process behind labeling.
3. The classifier is encouraged to adapt the provided examples as guidelines without labeling them.
4. The script generates responses based on the chain of thought, helping streamline the classification process.

By following this structured approach, the classifier can efficiently categorize Slovenian text while maintaining consistency and transparency in the decision-making process.

## Classes and Functions

- **AutoTokenizer:** Tokenizes input text for model compatibility.
- **AutoModelForCausalLM:** Loads the pre-trained language model for generation.
- **BitsAndBytesConfig:** Configuration for model quantization.
- **GenerationConfig:** Configuration for text generation.
- **label(texts):** Function for classifying Slovenian text based on predefined prompts.

Note: The script assumes the existence of a CSV file named "violence.csv" containing the relevant data for classification.

Feel free to adapt and integrate this script into your workflow for Slovenian text classification tasks.
