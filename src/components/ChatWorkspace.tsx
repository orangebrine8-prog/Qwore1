import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowUp,
  Code2,
  Download,
  Film,
  FlaskConical,
  Image as ImageIcon,
  KeyRound,
  MessageSquare,
  RotateCcw,
  SlidersHorizontal,
  Square,
} from 'lucide-react';
import type { MultimodalCheckpoint } from '../engine/multimodal';
import type { QoweCheckpoint } from '../engine/trainer';
import type {
  ChatEngineMode,
  ChatMessage,
  ChatOutputMode,
  Conversation,
  ReasoningDepth,
  SamplingSettings,
  WorkspaceTab,
} from '../types/qowe';
import { triggerDataUrlDownload } from '../utils/mediaExport';
import { CanvasMediaPlayer } from './CanvasMediaPlayer';
import { LiveSceneVideoPlayer } from './LiveSceneVideoPlayer';
import { MarkdownMessage } from './MarkdownMessage';

interface ChatWorkspaceProps {
  conversation: Conversation;
  engineMode: ChatEngineMode;
  onChangeEngineMode: (mode: ChatEngineMode) => void;
  reasoningDepth: ReasoningDepth;
  onChangeReasoningDepth: (depth: ReasoningDepth) => void;
  outputMode: ChatOutputMode;
  onChangeOutputMode: (mode: ChatOutputMode) => void;
  videoEngineSource: 'scratch-neural' | 'scene-script';
  onChangeVideoEngineSource: (source: 'scratch-neural' | 'scene-script') => void;
  sampling: SamplingSettings;
  onChangeSampling: (next: SamplingSettings) => void;
  isStreaming: boolean;
  onSendMessage: (text: string, overrideOutputMode?: ChatOutputMode) => void;
  onStopStreaming: () => void;
  onClearConversation: () => void;
  onRetryLast: () => void;
  architectureName: string;
  totalParamsFormatted: string;
  checkpoint: QoweCheckpoint | null;
  multimodalCheckpoint: MultimodalCheckpoint | null;
  onQuickTrainMultimodal: () => void;
  isQuickTrainingMultimodal: boolean;
  onOpenDownloadModal: () => void;
  onNavigateTab: (tab: WorkspaceTab) => void;
  onOpenMobileMenu: () => void;
}

const STARTER_PROMPTS: {
  title: string;
  prompt: string;
  outputMode?: ChatOutputMode;
}[] = [
  {
    title: 'Build a Full-Stack SaaS Tracker',
    prompt: 'Build a full-stack project & sprint issue tracker with Express REST API, SQL schema with seed issues, live search/status filters, and persistent database storage.',
    outputMode: 'build',
  },
  {
    title: 'Generate 60 FPS Video',
    prompt: 'neon cyber city rain reflections at night with sleek flying vehicles',
    outputMode: 'video',
  },
  {
    title: 'Explain Transformer Self-Attention',
    prompt: 'Explain how causal multi-head self-attention works inside Qowe 1.0 step by step.',
    outputMode: 'text',
  },
  {
    title: 'Generate Studio Image',
    prompt: 'studio golden banana sculpture resting on dark brushed slate with dramatic rim lighting',
    outputMode: 'image',
  },
];

interface ChatMessageItemProps {
  msg: ChatMessage;
  isLast: boolean;
  isStreamingMsg: boolean;
  onRetryLast: () => void;
}

const ChatMessageItem: React.FC<ChatMessageItemProps> = React.memo(
  ({ msg, isLast, isStreamingMsg, onRetryLast }) => {
    const isUser = msg.role === 'user';

    if (isUser) {
      return (
        <div className="flex justify-end">
          <div className="max-w-[85%] sm:max-w-[75%] px-4 py-3 rounded-[22px] bg-[#007aff] dark:bg-[#0a84ff] text-white text-[15px] leading-relaxed whitespace-pre-wrap shadow-xs">
            {msg.content}
          </div>
        </div>
      );
    }

    return (
      <div className="flex justify-start">
        <div className="w-full space-y-2">
          <div className="flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400 px-1">
            <span className="font-semibold text-neutral-900 dark:text-white">Qowe 1.0</span>
            <span aria-hidden="true">·</span>
            <span>
              {msg.imageDataUrl || msg.svgMarkup || msg.videoScene || msg.videoData
                ? 'Multimodal Studio'
                : msg.modeUsed === 'local-weights'
                  ? 'Local Checkpoint'
                  : msg.reasoningDepth === 'deep'
                    ? 'Deep Reasoning'
                    : 'Assistant'}
            </span>
            {typeof msg.tokenCount === 'number' && msg.tokenCount > 0 && (
              <>
                <span aria-hidden="true">·</span>
                <span className="tabular-nums">{msg.tokenCount} tokens</span>
              </>
            )}
          </div>

          {msg.error ? (
            <div className="p-4 rounded-[22px] border border-red-200 dark:border-red-900/60 bg-red-50 dark:bg-red-950/30 flex items-start justify-between gap-3">
              <div className="flex items-start gap-2.5 text-sm text-red-800 dark:text-red-200">
                <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
                <span>{msg.error}</span>
              </div>
              {isLast && (
                <button
                  type="button"
                  onClick={onRetryLast}
                  className="px-3.5 py-1.5 rounded-full bg-red-700 hover:bg-red-800 text-white text-xs font-medium whitespace-nowrap shrink-0"
                >
                  Retry
                </button>
              )}
            </div>
          ) : (
            <div className="bg-white dark:bg-[#1c1c1e] border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] rounded-[22px] px-5 py-4">
              {msg.content ? (
                <MarkdownMessage content={msg.content} isStreaming={isStreamingMsg} />
              ) : (
                <span className="text-sm text-neutral-400">Generating response...</span>
              )}

              {(msg.imageDataUrl || msg.svgMarkup) && (
                <div className="mt-3 space-y-2.5">
                  <div className="inline-block rounded-2xl overflow-hidden border border-black/[0.06] dark:border-white/[0.08] bg-neutral-950 max-w-full">
                    <img
                      src={
                        msg.imageDataUrl ||
                        `data:image/svg+xml;charset=utf-8,${encodeURIComponent(msg.svgMarkup || '')}`
                      }
                      alt="Generated by Qowe 1.0"
                      loading="lazy"
                      className="w-72 sm:w-96 max-w-full h-auto object-contain block"
                    />
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    {msg.imageDataUrl && (
                      <button
                        type="button"
                        onClick={() =>
                          triggerDataUrlDownload(
                            msg.imageDataUrl as string,
                            'qowe-1.0-image.png',
                          )
                        }
                        className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08]"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Download PNG</span>
                      </button>
                    )}
                    {msg.svgMarkup && (
                      <button
                        type="button"
                        onClick={() =>
                          triggerDataUrlDownload(
                            `data:image/svg+xml;charset=utf-8,${encodeURIComponent(msg.svgMarkup || '')}`,
                            'qowe-1.0-image.svg',
                          )
                        }
                        className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08]"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Download SVG</span>
                      </button>
                    )}
                  </div>
                </div>
              )}

              {msg.videoScene && (
                <div className="mt-3">
                  <LiveSceneVideoPlayer spec={msg.videoScene} />
                </div>
              )}

              {msg.videoData &&
                ((Array.isArray(msg.videoData.frames) && msg.videoData.frames.length > 0) ||
                  Boolean(msg.videoData.cinemaPayload)) && (
                  <div className="mt-3">
                    <CanvasMediaPlayer
                      frames={msg.videoData.frames ?? []}
                      width={msg.videoData.width}
                      height={msg.videoData.height}
                      fps={msg.videoData.fps}
                      audioTrack={msg.videoData.audioTrack}
                      cameraMotion={msg.videoData.cameraMotion}
                      cinemaPayload={msg.videoData.cinemaPayload}
                      showScrubber
                      label="Generated 60 fps LTX-2.5 3D PBR audio + video"
                    />
                  </div>
                )}
            </div>
          )}

          {msg.topCandidates && msg.topCandidates.length > 0 && (
            <div className="pt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-neutral-500 dark:text-neutral-400 font-mono tabular-nums px-1">
              <span>Next-token candidates:</span>
              {msg.topCandidates.map((cand) => (
                <span key={`${cand.id}-${cand.token}`}>
                  "{cand.token.replace(/\n/g, '\\n')}" {(cand.probability * 100).toFixed(1)}%
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  },
);

interface ChatComposerProps {
  outputMode: ChatOutputMode;
  onChangeOutputMode: (mode: ChatOutputMode) => void;
  videoEngineSource: 'scratch-neural' | 'scene-script';
  onChangeVideoEngineSource: (source: 'scratch-neural' | 'scene-script') => void;
  engineMode: ChatEngineMode;
  reasoningDepth: ReasoningDepth;
  onChangeReasoningDepth: (depth: ReasoningDepth) => void;
  isStreaming: boolean;
  localUntrained: boolean;
  onSendMessage: (text: string) => void;
  onStopStreaming: () => void;
}

const ChatComposer: React.FC<ChatComposerProps> = React.memo(
  ({
    outputMode,
    onChangeOutputMode,
    videoEngineSource,
    onChangeVideoEngineSource,
    engineMode,
    reasoningDepth,
    onChangeReasoningDepth,
    isStreaming,
    localUntrained,
    onSendMessage,
    onStopStreaming,
  }) => {
    const [draft, setDraft] = useState('');
    const textareaRef = useRef<HTMLTextAreaElement | null>(null);

    const resizeTextarea = useCallback((el: HTMLTextAreaElement | null) => {
      if (!el) return;
      el.style.height = 'auto';
      el.style.height = `${Math.min(180, Math.max(44, el.scrollHeight))}px`;
    }, []);

    const handleSubmit = (e?: React.FormEvent) => {
      e?.preventDefault();
      const trimmed = draft.trim();
      if (!trimmed || isStreaming) return;
      setDraft('');
      if (textareaRef.current) {
        textareaRef.current.style.height = '44px';
      }
      onSendMessage(trimmed);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    };

    return (
      <div className="px-4 sm:px-6 py-3.5 bg-white/95 dark:bg-[#161618]/95 border-t border-black/[0.06] dark:border-white/[0.08] shrink-0">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto space-y-2.5">
          {/* Output mode selector + reasoning indicator */}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div
              role="group"
              aria-label="Output format"
              className="flex flex-wrap items-center p-1 rounded-full bg-black/[0.05] dark:bg-white/[0.08]"
            >
              <button
                type="button"
                onClick={() => onChangeOutputMode('text')}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                  outputMode === 'text'
                    ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                    : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                }`}
              >
                <MessageSquare className="w-3.5 h-3.5" />
                <span>Chat</span>
              </button>
              <button
                type="button"
                onClick={() => onChangeOutputMode('build')}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                  outputMode === 'build'
                    ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                    : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                }`}
              >
                <Code2 className="w-3.5 h-3.5" />
                <span>Code &amp; Build</span>
              </button>
              <button
                type="button"
                onClick={() => onChangeOutputMode('image')}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                  outputMode === 'image'
                    ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                    : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                }`}
              >
                <ImageIcon className="w-3.5 h-3.5" />
                <span>Image</span>
              </button>
              <button
                type="button"
                onClick={() => onChangeOutputMode('video')}
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                  outputMode === 'video'
                    ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                    : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                }`}
              >
                <Film className="w-3.5 h-3.5" />
                <span>Video (60 fps)</span>
              </button>
            </div>

            {(outputMode === 'text' || outputMode === 'build') && engineMode === 'assistant' && (
              <div
                role="group"
                aria-label="Reasoning depth"
                className="flex items-center p-1 rounded-full bg-black/[0.05] dark:bg-white/[0.08]"
              >
                <button
                  type="button"
                  onClick={() => onChangeReasoningDepth('standard')}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                    reasoningDepth === 'standard'
                      ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                  }`}
                >
                  Standard
                </button>
                <button
                  type="button"
                  onClick={() => onChangeReasoningDepth('deep')}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                    reasoningDepth === 'deep'
                      ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                  }`}
                >
                  Deep Reasoning
                </button>
              </div>
            )}

            {outputMode === 'video' && (
              <div
                role="group"
                aria-label="Video engine source"
                className="flex items-center p-1 rounded-full bg-black/[0.05] dark:bg-white/[0.08]"
              >
                <button
                  type="button"
                  onClick={() => onChangeVideoEngineSource('scratch-neural')}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                    videoEngineSource === 'scratch-neural' || engineMode === 'local-weights'
                      ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                  }`}
                >
                  LTX-2.5 3D (Audio + 60fps)
                </button>
                <button
                  type="button"
                  disabled={engineMode === 'local-weights'}
                  onClick={() => onChangeVideoEngineSource('scene-script')}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap disabled:opacity-45 ${
                    videoEngineSource === 'scene-script' && engineMode !== 'local-weights'
                      ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
                  }`}
                >
                  Scene Script
                </button>
              </div>
            )}
          </div>

          <div className="flex items-end gap-2 p-2 rounded-[26px] border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] focus-within:border-[#007aff] dark:focus-within:border-[#0a84ff] transition-colors">
            <label htmlFor="qowe-chat-input" className="sr-only">
              Message Qowe 1.0
            </label>
            <textarea
              id="qowe-chat-input"
              ref={textareaRef}
              rows={1}
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value);
                resizeTextarea(e.currentTarget);
              }}
              onKeyDown={handleKeyDown}
              placeholder={
                outputMode === 'build'
                  ? 'Describe a full-stack app (Frontend + Backend API + SQL Database) to architect, run live & export .zip...'
                  : outputMode === 'image'
                    ? 'Describe an image for Qowe 1.0 to generate...'
                    : outputMode === 'video'
                      ? 'Describe a scene for Qowe 1.0 to animate at 60 fps...'
                      : engineMode === 'local-weights'
                        ? 'Prompt your locally trained Qowe 1.0 weights...'
                        : 'Ask Qowe 1.0 anything (chat, code, image, or video)...'
              }
              className="flex-1 resize-none bg-transparent px-3.5 py-2 text-sm text-neutral-900 dark:text-neutral-100 placeholder:text-neutral-400 focus:outline-none max-h-44"
            />
            {isStreaming ? (
              <button
                type="button"
                onClick={onStopStreaming}
                aria-label="Stop generation"
                className="p-2.5 rounded-full bg-neutral-900 dark:bg-white text-white dark:text-neutral-900 hover:opacity-90 transition-opacity shrink-0"
              >
                <Square className="w-4 h-4 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!draft.trim()}
                aria-label="Send message"
                className="p-2.5 rounded-full bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] disabled:bg-black/[0.08] dark:disabled:bg-white/[0.08] disabled:text-neutral-400 text-white transition-colors shrink-0"
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            )}
          </div>
          <p className="text-center text-xs text-neutral-400 dark:text-neutral-500">
            Press Enter to send, Shift+Enter for a new line.
          </p>
        </form>
      </div>
    );
  },
);

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  conversation,
  engineMode,
  onChangeEngineMode,
  reasoningDepth,
  onChangeReasoningDepth,
  outputMode,
  onChangeOutputMode,
  videoEngineSource,
  onChangeVideoEngineSource,
  sampling,
  onChangeSampling,
  isStreaming,
  onSendMessage,
  onStopStreaming,
  onClearConversation,
  onRetryLast,
  architectureName,
  totalParamsFormatted,
  checkpoint,
  onOpenDownloadModal,
  onNavigateTab,
}) => {
  const [showSampling, setShowSampling] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const stickToBottomRef = useRef(true);
  const prevMsgCountRef = useRef(conversation.messages.length);

  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distanceFromBottom < 140;
  }, []);

  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const msgCountChanged = conversation.messages.length !== prevMsgCountRef.current;
    prevMsgCountRef.current = conversation.messages.length;
    if (msgCountChanged) {
      stickToBottomRef.current = true;
    }
    if (stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [conversation.messages, conversation.id]);

  const hasMessages = conversation.messages.length > 0;
  const localUntrained = outputMode === 'text' && engineMode === 'local-weights' && !checkpoint;

  return (
    <div className="flex flex-col flex-1 h-full min-h-0 min-w-0 bg-[#f5f5f7] dark:bg-[#0b0b0d]">
      {/* Top workspace bar */}
      <header className="flex items-center justify-between gap-2 sm:gap-3 px-3 sm:px-6 py-2.5 sm:py-3 bg-white/95 dark:bg-[#161618]/95 border-b border-black/[0.06] dark:border-white/[0.08] shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <div className="min-w-0">
            <h1 className="text-sm sm:text-base font-semibold text-neutral-900 dark:text-white truncate">
              {architectureName}
            </h1>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 truncate tabular-nums">
              {totalParamsFormatted} parameters
              <span aria-hidden="true"> · </span>
              {outputMode === 'image'
                ? 'Studio Image Generation'
                : outputMode === 'video'
                  ? '60 FPS Motion Video Generation'
                  : outputMode === 'build'
                    ? 'Full-Stack App Builder (Frontend + API + SQL)'
                    : engineMode === 'assistant'
                      ? `Smart Assistant (${reasoningDepth === 'deep' ? 'Deep Reasoning' : 'Standard'})`
                      : checkpoint
                        ? `Local Checkpoint (Step ${checkpoint.step})`
                        : 'Local Checkpoint (Not Trained)'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <div
            role="group"
            aria-label="Text inference engine mode"
            className="flex items-center p-1 rounded-full bg-black/[0.05] dark:bg-white/[0.08]"
          >
            <button
              type="button"
              onClick={() => {
                onChangeEngineMode('assistant');
                if (outputMode !== 'text') onChangeOutputMode('text');
              }}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                engineMode === 'assistant' && outputMode === 'text'
                  ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                  : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
              }`}
            >
              Assistant
            </button>
            <button
              type="button"
              onClick={() => {
                onChangeEngineMode('local-weights');
                if (outputMode !== 'text') onChangeOutputMode('text');
              }}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                engineMode === 'local-weights' && outputMode === 'text'
                  ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                  : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white'
              }`}
            >
              <span className="hidden sm:inline">Trained </span>Weights
            </button>
          </div>

          <button
            type="button"
            onClick={() => setShowSampling((prev) => !prev)}
            aria-label="Toggle reasoning and sampling controls"
            aria-expanded={showSampling}
            className={`p-2 rounded-full border text-xs transition-colors ${
              showSampling
                ? 'border-[#007aff] text-[#007aff] bg-[#007aff]/10 dark:border-[#0a84ff] dark:bg-[#0a84ff]/15 dark:text-[#0a84ff]'
                : 'border-black/[0.08] dark:border-white/[0.1] text-neutral-600 dark:text-neutral-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.08]'
            }`}
          >
            <SlidersHorizontal className="w-4 h-4" />
          </button>

          <button
            type="button"
            onClick={() => onNavigateTab('api')}
            aria-label="Open Qowe 1.0 API Keys"
            className="hidden sm:inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] transition-colors whitespace-nowrap shrink-0"
          >
            <KeyRound className="w-3.5 h-3.5" />
            <span>API Keys</span>
          </button>

          <button
            type="button"
            onClick={onOpenDownloadModal}
            aria-label="Download Qowe 1.0 model"
            className="hidden sm:inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] transition-colors whitespace-nowrap shrink-0"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export</span>
          </button>

          {hasMessages && (
            <button
              type="button"
              onClick={onClearConversation}
              aria-label="Clear current conversation"
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.04] dark:hover:bg-white/[0.08] transition-colors whitespace-nowrap shrink-0"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Clear</span>
            </button>
          )}
        </div>
      </header>

      {/* Reasoning & Sampling controls bar */}
      {showSampling && (
        <div className="px-4 sm:px-6 py-3 bg-white dark:bg-[#1c1c1e] border-b border-black/[0.06] dark:border-white/[0.08]">
          <div className="max-w-3xl mx-auto grid grid-cols-1 sm:grid-cols-4 gap-4 items-center">
            <div className="text-xs">
              <div className="font-medium text-neutral-800 dark:text-neutral-200 mb-1">
                Assistant Reasoning
              </div>
              <div className="flex items-center p-0.5 rounded-full bg-black/[0.05] dark:bg-white/[0.08]">
                <button
                  type="button"
                  onClick={() => onChangeReasoningDepth('standard')}
                  className={`flex-1 px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                    reasoningDepth === 'standard'
                      ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400'
                  }`}
                >
                  Standard
                </button>
                <button
                  type="button"
                  onClick={() => onChangeReasoningDepth('deep')}
                  className={`flex-1 px-2.5 py-1 rounded-full text-xs font-medium transition-colors ${
                    reasoningDepth === 'deep'
                      ? 'bg-white dark:bg-[#2c2c2e] text-neutral-900 dark:text-white shadow-xs'
                      : 'text-neutral-600 dark:text-neutral-400'
                  }`}
                >
                  Deep
                </button>
              </div>
            </div>

            <label className="block text-xs text-neutral-600 dark:text-neutral-400">
              <div className="flex justify-between mb-1">
                <span className="font-medium text-neutral-800 dark:text-neutral-200">Temperature</span>
                <span className="font-mono tabular-nums">{sampling.temperature.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.5"
                step="0.05"
                value={sampling.temperature}
                onChange={(e) =>
                  onChangeSampling({ ...sampling, temperature: Number(e.target.value) })
                }
                className="w-full accent-[#007aff]"
              />
            </label>

            <label className="block text-xs text-neutral-600 dark:text-neutral-400">
              <div className="flex justify-between mb-1">
                <span className="font-medium text-neutral-800 dark:text-neutral-200">Top-K</span>
                <span className="font-mono tabular-nums">{sampling.topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="40"
                step="1"
                value={sampling.topK}
                onChange={(e) => onChangeSampling({ ...sampling, topK: Number(e.target.value) })}
                className="w-full accent-[#007aff]"
              />
            </label>

            <label className="block text-xs text-neutral-600 dark:text-neutral-400">
              <div className="flex justify-between mb-1">
                <span className="font-medium text-neutral-800 dark:text-neutral-200">Max Tokens</span>
                <span className="font-mono tabular-nums">{sampling.maxTokens}</span>
              </div>
              <input
                type="range"
                min="16"
                max="160"
                step="8"
                value={sampling.maxTokens}
                onChange={(e) =>
                  onChangeSampling({ ...sampling, maxTokens: Number(e.target.value) })
                }
                className="w-full accent-[#007aff]"
              />
            </label>
          </div>
        </div>
      )}

      {/* Main chat scroll area */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-4 sm:px-6 py-6"
      >
        <div className="max-w-3xl mx-auto">
          {localUntrained && (
            <div className="mb-6 p-4 rounded-[22px] border border-amber-300/60 dark:border-amber-800/60 bg-amber-50/90 dark:bg-amber-950/30 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-amber-900 dark:text-amber-200">
                  Local from-scratch weights ready to auto-train
                </p>
                <p className="text-xs text-amber-800 dark:text-amber-300">
                  Send any message to auto-train Qowe 1.0 locally from scratch on your dataset, or customize steps in Training Studio.
                </p>
              </div>
              <button
                type="button"
                onClick={() => onNavigateTab('train')}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-900 hover:bg-amber-800 dark:bg-amber-600 dark:hover:bg-amber-500 text-white text-xs font-medium transition-colors whitespace-nowrap shrink-0"
              >
                <FlaskConical className="w-3.5 h-3.5" />
                <span>Open Training Studio</span>
              </button>
            </div>
          )}

          {!hasMessages ? (
            <div className="py-4 sm:py-14 space-y-5 sm:space-y-8">
              <div className="space-y-2 sm:space-y-3">
                <h2 className="text-xl sm:text-3xl font-bold tracking-tight text-neutral-900 dark:text-white text-balance">
                  Chat, Reason &amp; Generate with Qowe 1.0
                </h2>
                <p className="text-xs sm:text-base text-neutral-600 dark:text-neutral-400 max-w-2xl leading-relaxed">
                  Configured at {totalParamsFormatted} parameters as a multimodal transformer. Ask naturally to chat, build runnable apps, generate studio images, animate 60 fps videos, or create Qowe API keys for external apps.
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {STARTER_PROMPTS.map((item) => (
                  <button
                    key={item.title}
                    type="button"
                    onClick={() => {
                      if (item.outputMode) onChangeOutputMode(item.outputMode);
                      onSendMessage(item.prompt, item.outputMode);
                    }}
                    className="text-left p-4 sm:p-5 rounded-[22px] bg-white dark:bg-[#1c1c1e] border border-black/[0.06] dark:border-white/[0.08] shadow-[0_1px_2px_rgba(0,0,0,0.03),0_8px_24px_rgba(0,0,0,0.03)] hover:border-[#007aff]/60 dark:hover:border-[#0a84ff]/60 transition-colors space-y-1.5"
                  >
                    <div className="text-sm font-semibold text-neutral-900 dark:text-white">
                      {item.title}
                    </div>
                    <div className="text-xs text-neutral-500 dark:text-neutral-400 line-clamp-2">
                      {item.prompt}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-6" data-testid="chat-message-list">
              {conversation.messages.map((msg: ChatMessage, idx) => {
                const isLast = idx === conversation.messages.length - 1;
                return (
                  <ChatMessageItem
                    key={msg.id}
                    msg={msg}
                    isLast={isLast}
                    isStreamingMsg={isStreaming && isLast && msg.role === 'assistant'}
                    onRetryLast={onRetryLast}
                  />
                );
              })}
            </div>
          )}
        </div>
      </div>

      <ChatComposer
        outputMode={outputMode}
        onChangeOutputMode={onChangeOutputMode}
        videoEngineSource={videoEngineSource}
        onChangeVideoEngineSource={onChangeVideoEngineSource}
        engineMode={engineMode}
        reasoningDepth={reasoningDepth}
        onChangeReasoningDepth={onChangeReasoningDepth}
        isStreaming={isStreaming}
        localUntrained={localUntrained}
        onSendMessage={onSendMessage}
        onStopStreaming={onStopStreaming}
      />
    </div>
  );
};
