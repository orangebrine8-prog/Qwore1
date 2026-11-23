export interface TrainingDatasetPreset {
  id: string;
  name: string;
  category: string;
  description: string;
  corpus: string;
}

export const TRAINING_DATASETS: TrainingDatasetPreset[] = [
  {
    id: 'conversational-qa',
    name: 'Qowe Conversational Dialogues',
    category: 'Chat & Instruction',
    description: 'Multi-turn assistant Q&A pairs covering identity, reasoning, multimodal synthesis, and code.',
    corpus: `<|user|> Hi
<|assistant|> Hello! I am Qowe 1.0, your local transformer assistant. Ask me about my architecture, training, code, or multimodal generation.<|endoftext|>
<|user|> What can you do?
<|assistant|> I can answer questions about transformers and neural networks, help write clean code, and drive from-scratch image and 60 fps video synthesis.<|endoftext|>
<|user|> Who are you?
<|assistant|> I am Qowe 1.0, a 124.44 million parameter causal transformer trained from scratch to reason clearly and generate text, images, and 60 fps video.<|endoftext|>
<|user|> Hello Qowe! How do you work?
<|assistant|> Hello! I tokenize your prompt, apply causal multi-head self-attention across transformer layers, and predict tokens or spatiotemporal pixels step by step.<|endoftext|>
<|user|> What is a transformer model?
<|assistant|> A transformer is a neural network architecture that uses scaled dot-product self-attention to weigh relationships between all tokens in a context window.<|endoftext|>
<|user|> How many parameters does Qowe 1.0 have?
<|assistant|> Qowe 1.0 Base has 124,439,808 parameters across 12 decoder layers, 12 attention heads, and a 768 hidden dimension.<|endoftext|>
<|user|> How does your 60 fps video model work?
<|assistant|> My spatiotemporal neural field conditions hidden layers on your text embedding using FiLM scale and shift, then evaluates continuous coordinates (x, y, t) at 60 frames per second.<|endoftext|>
<|user|> Why is causal masking important?
<|assistant|> Causal masking prevents each token from attending to future tokens during training, ensuring the model learns genuine left-to-right next-token prediction.<|endoftext|>
<|user|> Explain gradient descent simply.
<|assistant|> Gradient descent iteratively updates model weights in the opposite direction of the loss gradient using AdamW to minimize prediction error.<|endoftext|>
<|user|> What is perplexity?
<|assistant|> Perplexity is the exponent of cross-entropy loss and measures how uncertain a language model is when predicting the next token.<|endoftext|>
<|user|> Can you help me write clean code?
<|assistant|> Yes! Share your programming task or architecture question, and I will break down the logic and write clean, structured code.<|endoftext|>`,
  },
  {
    id: 'python-algorithms',
    name: 'Python & TypeScript Algorithms',
    category: 'Code Synthesis',
    description: 'Clean algorithmic implementations for sorting, search, recursion, and neural math.',
    corpus: `<|user|> Write binary search in Python.
<|assistant|> def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1<|endoftext|>
<|user|> Write softmax in TypeScript.
<|assistant|> function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}<|endoftext|>
<|user|> Write a Fibonacci generator in Python.
<|assistant|> def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b<|endoftext|>`,
  },
  {
    id: 'classical-prose',
    name: 'Scientific & Literary Prose',
    category: 'Language Modeling',
    description: 'Dense philosophical and scientific paragraphs for testing next-token language modeling.',
    corpus: `Every computation begins with structure. When tokens enter an embedding matrix, discrete symbols become continuous vectors in high-dimensional space. Across layers of attention and nonlinear projection, patterns emerge from raw text without hand-coded grammar rules.<|endoftext|>
Observation precedes theory, yet theory guides what we know how to observe. A careful experiment measures not only the signal we expect, but the residuals that reveal where our assumptions break down.<|endoftext|>
In the quiet hours before dawn, the observatory dome rotated toward the eastern horizon, tracking faint light that had crossed millions of years of empty space before striking the sensor.<|endoftext|>`,
  },
];
