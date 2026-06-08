import openai

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import pandas as pd

from langchain.schema import HumanMessage
import time
import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema import HumanMessage
import pandas as pd
import os
import time


# loading models
API_KEY=''
client = openai.OpenAI(
    api_key = API_KEY,
    base_url="https://test-llm.rdc.nie.edu.sg/api/v1",
)

qwen3AWQ = ChatOpenAI(
    model_name="Qwen3-32B-AWQ",
    openai_api_key="",
    openai_api_base="https://test-llm.rdc.nie.edu.sg/api/v1",
    temperature=0.9,           
)

deepseek = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_key="",
    openai_api_base="https://api.deepseek.com",
    temperature=0.9)



gpt4o_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.9,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=API_KEY
)

# data cleaning/engineering

df=pd.read_excel(r'TeacherGAIA_chatlog_filtered_Deidentified.xlsx')

df['prompt'] = df['prompt'].str.replace('\n', ' ').str.strip()
df['response'] = df['response'].str.replace('\n', ' ').str.strip()
df=df.drop(columns=['id'])
df=df.dropna()




def adding_features(df):
    df = df.sort_values(['studId', 'timestamp'])

    df['previous_context'] = df.groupby('studId').apply(
        lambda x: 'Previous Q: ' + x['prompt'].shift(1) + ' | AI BOT said: ' + x['response'].shift(1)
    ).reset_index(level=0, drop=True)

    df['previous_context'] = df['previous_context'].fillna('conversation_starter')

    return df

df=adding_features(df)


ask_examples_lines = [
    {
        "prompt": "what should i eat today?",
        "output": """basic_question: 0.7
analytical_question: 0.0
complex_analytical: 0.0
probing_deeper: 0.0
non_question: 0.3"""
    },
    {
        "prompt": "What are some examples of challenges for managing differing needs and interests in the context of both Singapore and Korea",
        "output": """basic_question: 0.0
analytical_question: 0.1
complex_analytical: 0.8
probing_deeper: 0.0
non_question: 0.1"""
    },
    {
        "prompt": "Why do you think so",
        "output": """basic_question: 0.0
analytical_question: 0.1
complex_analytical: 0.0
probing_deeper: 0.9
non_question: 0.0"""
    },
    {
        "prompt": "😊",
        "output": """basic_question: 0.0
analytical_question: 0.0
complex_analytical: 0.0
probing_deeper: 0.0
non_question: 1.0"""
    },
    {
        "prompt": "What makes it hard for the government to manage unequal sharing of costs?",
        "output": """basic_question: 0.0
analytical_question: 0.8
complex_analytical: 0.1
probing_deeper: 0.0
non_question: 0.1"""
    }
]


buildupon_examples_lines = [
    {
        "prompt": "recipe for kimchi",
        "context": "Previous Q: 'what should i eat today?' | AI BOT said: 'As a teacher of Korean history and culture, I might suggest that you try a dish from Korea to immerse yourself in that culture. For instance, you could look up a recipe for kimchi, a traditional and healthy Korean fermented dish. It's both interesting and educational to explore the tastes and cooking methods of different cultures",
        "output": """initial_inquiry: 0.0
direct_followup: 0.9
clarification_request: 0.0
expansion_request: 0.1
acknowledgment_agreement: 0.0
topic_shift: 0.0"""
    },
    {
        "prompt": "How should I describe how sad he feels?",
        "context": "Previous Q: 'I would like some revision advice, especially about the character’s feelings! \"I argued with Xiaowen and felt very sad.\"' | AI BOT said: 'You can try to describe the character’s emotions more deeply. For example, how sad does he feel? What does his pain feel like? How does he show his sadness?' ",
        "output": """initial_inquiry: 0.0
direct_followup: 0.1
clarification_request: 0.8
expansion_request: 0.1
acknowledgment_agreement: 0.0
topic_shift: 0.0"""
    },
    {
        "prompt": "another example please",
        "context": "Previous Q: 'example of unequal sharing of costs between citizens in singapore' | AI BOT said: 'One example is the way Singapore's progressive tax...'",
        "output": """initial_inquiry: 0.0
direct_followup: 0.1
clarification_request: 0.0
expansion_request: 0.9
acknowledgment_agreement: 0.0
topic_shift: 0.0"""
    },
    {
        "prompt": "what should i eat today?",
        "context": "conversation starter",
        "output": """initial_inquiry: 1.0
direct_followup: 0.0
clarification_request: 0.0
expansion_request: 0.0
acknowledgment_agreement: 0.0
topic_shift: 0.0"""
    },
    {
        "prompt": "That makes sense.",
        "context": "Previous Q: 'At 11 p.m. one night, Xiaowen was still playing computer games and had not gone to sleep. His mother saw it and was very angry. Is this sentence coherent?' | AI BOT said: 'Your sentence is already easy to understand. However, could we describe more clearly why the mother was angry? For example, was she worried about Xiaowen’s health, or angry because he broke the rule about bedtime? How did the mother show her anger (e.g., frowning, cold tone)?'",
        "output": """initial_inquiry: 0.0
direct_followup: 0.1
clarification_request: 0.0
expansion_request: 0.0
acknowledgment_agreement: 0.9
topic_shift: 0.0"""
    }
]


critical_thinking_examples_lines = [
    {
        "prompt": "Do community groups have a more important role to play than the government in working together to address the needs of the low-income groups in Singapore? Explain your answer.",
        "output": """factual_inquiry: 0.0
questioning_reasoning: 0.1
comparative_evaluation: 0.8
causal_analysis: 0.0
evaluative_judgment: 0.1
synthesis_integration: 0.0"""
    },
    {
        "prompt": "Why do you think so",
        "output": """factual_inquiry: 0.0
questioning_reasoning: 0.9
comparative_evaluation: 0.0
causal_analysis: 0.1
evaluative_judgment: 0.0
synthesis_integration: 0.0"""
    },
    {
        "prompt": "is the governmnet's efforts to help the low income groups in vain? why/ why not",
        "output": """factual_inquiry: 0.0
questioning_reasoning: 0.1
comparative_evaluation: 0.0
causal_analysis: 0.1
evaluative_judgment: 0.8
synthesis_integration: 0.0"""
    },
    {
        "prompt": "basic greetings in korea",
        "output": """factual_inquiry: 1.0
questioning_reasoning: 0.0
comparative_evaluation: 0.0
causal_analysis: 0.0
evaluative_judgment: 0.0
synthesis_integration: 0.0"""
    },
    {
        "prompt": "how this inclusive approach creates a positive cycle of change in the society?",
        "output": """factual_inquiry: 0.0
questioning_reasoning: 0.0
comparative_evaluation: 0.0
causal_analysis: 0.9
evaluative_judgment: 0.0
synthesis_integration: 0.1"""
    }
]


ask_template_lines = PromptTemplate(
    input_variables=["prompt", "output"],
    template="""Student Prompt: {prompt}
Output:
{output}"""
)

buildupon_template_lines = PromptTemplate(
    input_variables=["prompt", "context", "output"],
    template="""Student Prompt: {prompt}
Context: {context}
Output:
{output}"""
)

critical_thinking_template_lines = PromptTemplate(
    input_variables=["prompt", "output"],
    template="""Student Prompt: {prompt}
Output:
{output}"""
)

#### 


ask_few_shot_prompt_lines = FewShotPromptTemplate(
    examples=ask_examples_lines,
    example_prompt=ask_template_lines,
    prefix="""You are an expert in analyzing student learning interactions. Your task is to classify student prompts based on their questioning behavior.

Classification Categories:
- basic_question: Simple, direct questions seeking factual information
- analytical_question: Questions requiring analysis of causes, effects, or mechanisms  
- complex_analytical: Multi-faceted questions requiring comparison, synthesis, or deep analysis
- probing_deeper: Follow-up questions seeking more detailed explanations or reasoning
- non_question: Statements, expressions, or non-questioning prompts

STRICT OUTPUT RULES:
1. Output EXACTLY 5 lines, one per category, in this exact order
2. Format: category_name: probability (no spaces around colon)
3. Probabilities must be numbers between 0 and 1
4. Probabilities must sum to 1.0
5. NO extra text, explanations, markdown, asterisks, or additional lines
6. Start immediately with first category

Examples:
""",
    suffix="""
Student Prompt: {input}

Output:""",
    input_variables=["input"]
)

buildupon_few_shot_prompt_lines = FewShotPromptTemplate(
    examples=buildupon_examples_lines,
    example_prompt=buildupon_template_lines,
    prefix="""You are an expert in analyzing student learning interactions. Your task is to classify how students build upon previous interactions.

Classification Categories:
- initial_inquiry: First question in conversation, not building on anything
- direct_followup: Directly builds on chatbot's previous response or suggestion
- clarification_request: Asks for clarification or more specific guidance on previous topic
- expansion_request: Requests additional examples, details, or related information
- acknowledgment_agreement: Acknowledges, agrees with, or expresses understanding of previous response
- topic_shift: Changes subject without building on previous interaction

STRICT OUTPUT RULES:
1. Output EXACTLY 6 lines, one per category, in this exact order
2. Format: category_name: probability (no spaces around colon)
3. Probabilities must be numbers between 0 and 1
4. Probabilities must sum to 1.0
5. NO extra text, explanations, markdown, asterisks, or additional lines
6. Start immediately with first category

Examples:
""",
    suffix="""
Student Prompt: {input}
Context: {context}

Output:""",
    input_variables=["input", "context"]
)

critical_thinking_few_shot_prompt_lines = FewShotPromptTemplate(
    examples=critical_thinking_examples_lines,
    example_prompt=critical_thinking_template_lines,
    prefix="""You are an expert in analyzing student learning interactions. Your task is to classify the level of critical thinking demonstrated in student prompts.

Classification Categories:
- factual_inquiry: Simple requests for factual information, no analysis required
- questioning_reasoning: Challenges assumptions, seeks justification, or questions reasoning
- comparative_evaluation: Compares options, weighs alternatives, evaluates relative merits
- causal_analysis: Analyzes cause-and-effect relationships, mechanisms, or processes
- evaluative_judgment: Makes judgments about effectiveness, value, or worth with reasoning
- synthesis_integration: Combines multiple ideas, concepts, or perspectives into new understanding

STRICT OUTPUT RULES:
1. Output EXACTLY 6 lines, one per category, in this exact order
2. Format: category_name: probability (no spaces around colon)
3. Probabilities must be numbers between 0 and 1
4. Probabilities must sum to 1.0
5. NO extra text, explanations, markdown, asterisks, or additional lines
6. Start immediately with first category

Examples:
""",
    suffix="""
Student Prompt: {input}

Output:""",
    input_variables=["input"]
)

### no reasoning


ask_few_shot_prompt_lines_no_think = FewShotPromptTemplate(
    examples=ask_examples_lines,
    example_prompt=ask_template_lines,
    prefix="""/no_think
You are an expert in analyzing student learning interactions. Your task is to classify student prompts based on their questioning behavior.

Classification Categories:
- basic_question: Simple, direct questions seeking factual information
- analytical_question: Questions requiring analysis of causes, effects, or mechanisms  
- complex_analytical: Multi-faceted questions requiring comparison, synthesis, or deep analysis
- probing_deeper: Follow-up questions seeking more detailed explanations or reasoning
- non_question: Statements, expressions, or non-questioning prompts

STRICT OUTPUT RULES:
1. Output EXACTLY 5 lines, one per category, in this exact order
2. Format: category_name: probability (no spaces around colon)
3. Probabilities must be numbers between 0 and 1
4. Probabilities must sum to 1.0
5. NO extra text, explanations, markdown, asterisks, or additional lines
6. Start immediately with first category

Examples:
""",
    suffix="""
Student Prompt: {input}

Output:""",
    input_variables=["input"]
)

buildupon_few_shot_prompt_lines_no_think = FewShotPromptTemplate(
    examples=buildupon_examples_lines,
    example_prompt=buildupon_template_lines,
    prefix="""/no_think
You are an expert in analyzing student learning interactions. Your task is to classify how students build upon previous interactions.

Classification Categories:
- initial_inquiry: First question in conversation, not building on anything
- direct_followup: Directly builds on chatbot's previous response or suggestion
- clarification_request: Asks for clarification or more specific guidance on previous topic
- expansion_request: Requests additional examples, details, or related information
- acknowledgment_agreement: Acknowledges, agrees with, or expresses understanding of previous response
- topic_shift: Changes subject without building on previous interaction

STRICT OUTPUT RULES:
1. Output EXACTLY 6 lines, one per category, in this exact order
2. Format: category_name: probability (no spaces around colon)
3. Probabilities must be numbers between 0 and 1
4. Probabilities must sum to 1.0
5. NO extra text, explanations, markdown, asterisks, or additional lines
6. Start immediately with first category

Examples:
""",
    suffix="""
Student Prompt: {input}
Context: {context}

Output:""",
    input_variables=["input", "context"]
)

critical_thinking_few_shot_prompt_lines_no_think = FewShotPromptTemplate(
    examples=critical_thinking_examples_lines,
    example_prompt=critical_thinking_template_lines,
    prefix="""/no_think
You are an expert in analyzing student learning interactions. Your task is to classify the level of critical thinking demonstrated in student prompts.

Classification Categories:
- factual_inquiry: Simple requests for factual information, no analysis required
- questioning_reasoning: Challenges assumptions, seeks justification, or questions reasoning
- comparative_evaluation: Compares options, weighs alternatives, evaluates relative merits
- causal_analysis: Analyzes cause-and-effect relationships, mechanisms, or processes
- evaluative_judgment: Makes judgments about effectiveness, value, or worth with reasoning
- synthesis_integration: Combines multiple ideas, concepts, or perspectives into new understanding

STRICT OUTPUT RULES:
1. Output EXACTLY 6 lines, one per category, in this exact order
2. Format: category_name: probability (no spaces around colon)
3. Probabilities must be numbers between 0 and 1
4. Probabilities must sum to 1.0
5. NO extra text, explanations, markdown, asterisks, or additional lines
6. Start immediately with first category

Examples:
""",
    suffix="""
Student Prompt: {input}

Output:""",
    input_variables=["input"]
)

####

class ChatbotClassifier:
    def __init__(self, model, reasoning=None, max_batch_size=20, max_workers=5):
        self.llm = model
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.reasoning = reasoning

        # Choose prompts based on reasoning mode
        if self.reasoning == 'off':
            self.ask_prompt = ask_few_shot_prompt_lines_no_think
            self.buildupon_prompt = buildupon_few_shot_prompt_lines_no_think
            self.critical_thinking_prompt = critical_thinking_few_shot_prompt_lines_no_think
        else:
            self.ask_prompt = ask_few_shot_prompt_lines
            self.buildupon_prompt = buildupon_few_shot_prompt_lines
            self.critical_thinking_prompt = critical_thinking_few_shot_prompt_lines

    @staticmethod
    def parse_label_probs(text):
        """Convert 'label: 0.42' lines to dict"""
        probs = {}
        for line in text.splitlines():
            if ':' in line:
                label, prob = line.split(':', 1)
                try:
                    probs[label.strip()] = float(prob.strip())
                except ValueError:
                    continue
        return probs

    def process_single(self, inp, prompt_template):
        """Process a single input safely"""
        try:
            message = HumanMessage(content=prompt_template.format(**inp))
            response = self.llm.invoke([message])
            return self.parse_label_probs(response.content.strip())
        except Exception as e:
            print(f"[Error] Input {inp}: {e}")
            return {}

    def _batch_process(self, prompt_template, inputs, input_keys=["input"], checkpoint_path=None):
        # --- Load checkpoint if exists ---
        results = []
        start_idx = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            df_ckpt = pd.read_csv(checkpoint_path)
            results = df_ckpt.to_dict('records')
            start_idx = len(results)
            print(f"[Resume] Starting from row {start_idx}")

        # --- Process in batches ---
        for i in range(start_idx, len(inputs), self.max_batch_size):
            batch_slice = inputs[i:i + self.max_batch_size]
            batch_inputs = [
                {k: v for k, v in zip(input_keys, item)} if isinstance(item, tuple)
                else {input_keys[0]: item}
                for item in batch_slice
            ]

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self.process_single, inp, prompt_template) for inp in batch_inputs]
                for future in as_completed(futures):
                    results.append(future.result())

            # --- Save checkpoint after each batch ---
            if checkpoint_path:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                print(f"[Checkpoint] Saved {len(results)}/{len(inputs)} results to {checkpoint_path}")

        return results

    # ---------- Classifiers ----------
    def A_classifier(self, texts, checkpoint_path=None):
        return self._batch_process(self.ask_prompt, texts, input_keys=["input"], checkpoint_path=checkpoint_path)

    def B_classifier(self, texts, contexts, checkpoint_path=None):
        combined_inputs = list(zip(texts, contexts))
        return self._batch_process(self.buildupon_prompt, combined_inputs, input_keys=["input", "context"], checkpoint_path=checkpoint_path)

    def C_classifier(self, texts, checkpoint_path=None):
        return self._batch_process(self.critical_thinking_prompt, texts, input_keys=["input"], checkpoint_path=checkpoint_path)


def apply_llm_to_dataset_result(model, original_df, reasoning=None, llm_name="model"):
    # Make folder for checkpoints
    os.makedirs(f'checkpoints/{llm_name}', exist_ok=True)

    start = time.time()
    df = original_df.copy()
    classifier = ChatbotClassifier(model=model, reasoning=reasoning, max_workers=5)

    df['A_classifier'] = classifier.A_classifier(df['prompt'], checkpoint_path=f'checkpoints/{llm_name}/check_A.csv')
    df['B_classifier'] = classifier.B_classifier(df['prompt'], df['previous_context'], checkpoint_path=f'checkpoints/{llm_name}/check_B.csv')
    df['C_classifier'] = classifier.C_classifier(df['prompt'], checkpoint_path=f'checkpoints/{llm_name}/check_C.csv')

    end = time.time()
    print(f"Execution time model {model}: {end - start:.2f} seconds")
    return df

#quen
quendf = apply_llm_to_dataset_result(qwen3AWQ, df, reasoning='off', llm_name='quen')

gptdf=apply_llm_to_dataset_result(gpt4o_llm,df,llm_name='gpt')

deepseekdf=apply_llm_to_dataset_result(deepseek,df,llm_name='deepseek')


