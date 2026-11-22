export interface BpeMerge {
  left: number;
  right: number;
  mergedId: number;
}

export interface QoweTokenizer {
  vocab: string[];
  tokenToId: Record<string, number>;
  merges: BpeMerge[];
  specialTokens: {
    pad: number;
    endoftext: number;
    user: number;
    assistant: number;
  };
  baseVocabSize: number;
}

export interface TokenPiece {
  id: number;
  text: string;
}

const SPECIAL_TOKENS = ['<|pad|>', '<|endoftext|>', '<|user|>', '<|assistant|>'] as const;

// GPT-style pre-tokenizer: splits special tokens, optional-leading-space words, numbers, punctuation, and newlines
const PRETOKEN_REGEX =
  /<\|pad\|>|<\|endoftext\|>|<\|user\|>|<\|assistant\|>| ?[A-Za-z0-9_]+| ?[^\sA-Za-z0-9_<|>]+|[<|>]+|\r?\n|[ \t]+/g;

export function pretokenize(text: string): string[] {
  const matches = text.match(PRETOKEN_REGEX);
  return matches ? matches : [];
}

export function trainTokenizer(corpus: string, targetVocabSize = 512): QoweTokenizer {
  const vocab: string[] = [...SPECIAL_TOKENS];
  const tokenToId: Record<string, number> = {};
  vocab.forEach((tok, idx) => {
    tokenToId[tok] = idx;
  });

  // 1. Add frequent whole pre-tokens (words, space+word, punctuation) from corpus so English words stay intact
  const freq = new Map<string, number>();
  for (const piece of pretokenize(corpus)) {
    if (tokenToId[piece] !== undefined) continue;
    freq.set(piece, (freq.get(piece) ?? 0) + 1);
  }

  const sortedPieces = Array.from(freq.entries()).sort((a, b) => b[1] - a[1]);
  const wordBudget = Math.max(64, targetVocabSize - 96);
  for (const [piece] of sortedPieces) {
    if (vocab.length >= wordBudget) break;
    if (tokenToId[piece] === undefined) {
      tokenToId[piece] = vocab.length;
      vocab.push(piece);
    }
  }

  // 2. Ensure single-character fallback coverage for essential ASCII & any characters in corpus
  const essentialChars =
    ' \n\t.,!?;:\'"()-_/+*=<>[]{}@#&0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
  for (const ch of essentialChars + corpus) {
    if (tokenToId[ch] === undefined) {
      tokenToId[ch] = vocab.length;
      vocab.push(ch);
    }
  }

  const baseVocabSize = vocab.length;

  return {
    vocab,
    tokenToId,
    merges: [],
    specialTokens: {
      pad: 0,
      endoftext: 1,
      user: 2,
      assistant: 3,
    },
    baseVocabSize,
  };
}

export function encode(tokenizer: QoweTokenizer, text: string): number[] {
  const result: number[] = [];
  const pieces = pretokenize(text);

  for (const piece of pieces) {
    const directId = tokenizer.tokenToId[piece];
    if (directId !== undefined) {
      result.push(directId);
      continue;
    }

    // Greedy longest-match fallback over known vocab tokens, then single-character fallback
    let idx = 0;
    while (idx < piece.length) {
      const ch = piece[idx];
      const charId = tokenizer.tokenToId[ch];
      if (charId !== undefined) {
        result.push(charId);
      } else {
        result.push(tokenizer.tokenToId[' '] ?? 0);
      }
      idx += 1;
    }
  }

  return result;
}

export function decode(tokenizer: QoweTokenizer, ids: number[]): string {
  return ids
    .map((id) => tokenizer.vocab[id] ?? '')
    .filter((tok) => tok !== '<|pad|>')
    .join('');
}

export function tokenizeWithPieces(tokenizer: QoweTokenizer, text: string): TokenPiece[] {
  const ids = encode(tokenizer, text);
  return ids.map((id) => ({
    id,
    text: tokenizer.vocab[id] ?? '',
  }));
}
