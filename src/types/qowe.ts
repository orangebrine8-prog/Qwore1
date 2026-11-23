import type { VideoSceneSpec } from '../engine/intentAndMedia';
import type { LtxCinemaPayload } from '../engine/ltx25DiT';
import type { CameraMotionMode } from '../engine/multimodal';
import type { TokenCandidate } from '../engine/trainer';
import type { NeuralAudioFrame } from '../utils/neuralAudioSynth';

export type WorkspaceTab = 'chat' | 'train' | 'vision' | 'architecture' | 'api';
export type ChatEngineMode = 'assistant' | 'local-weights';
export type ReasoningDepth = 'standard' | 'deep';
export type ChatOutputMode = 'text' | 'build' | 'image' | 'video';

export interface GeneratedVideoData {
  width: number;
  height: number;
  fps: number;
  frames: Uint8ClampedArray[];
  audioTrack?: NeuralAudioFrame[];
  cameraMotion?: CameraMotionMode;
  cinemaPayload?: LtxCinemaPayload;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: string;
  modeUsed?: ChatEngineMode;
  reasoningDepth?: ReasoningDepth;
  tokenCount?: number;
  topCandidates?: TokenCandidate[];
  imageDataUrl?: string;
  svgMarkup?: string;
  videoData?: GeneratedVideoData;
  videoScene?: VideoSceneSpec;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  updatedAt: string;
  messages: ChatMessage[];
}

export interface SamplingSettings {
  temperature: number;
  topK: number;
  maxTokens: number;
}
