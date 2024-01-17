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

## Classes and Functions

- **AutoTokenizer:** Tokenizes input text for model compatibility.
- **AutoModelForCausalLM:** Loads the pre-trained language model for generation.
- **BitsAndBytesConfig:** Configuration for model quantization.
- **GenerationConfig:** Configuration for text generation.
- **label(texts):** Function for classifying Slovenian text based on predefined prompts.

Note: The script assumes the existence of a CSV file named "violence.csv" containing the relevant data for classification.

Feel free to adapt and integrate this script into your workflow for Slovenian text classification tasks.