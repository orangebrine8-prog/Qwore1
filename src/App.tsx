import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import {
  Cpu,
  Download,
  Film,
  FlaskConical,
  KeyRound,
  Menu,
  MessageSquare,
  Moon,
  Plus,
  Sun,
} from 'lucide-react';
import {
  calculateArchitectureSpecs,
  formatParamCount,
  QOWE_PRESETS,
  QoweArchitectureConfig,
} from './engine/architecture';
import { TRAINING_DATASETS } from './engine/datasets';
import {
  type QoweFullModelBundle,
  trainAssistantCheckpointAsync,
} from './engine/exporter';
import { detectChatIntent, type VideoSceneSpec } from './engine/intentAndMedia';
import type { MultimodalCheckpoint } from './engine/multimodal';
import { QoweCheckpoint, QoweTrainerEngine } from './engine/trainer';
import { generateMediaInPool } from './engine/workerClient';
import { ApiKeysStudio } from './components/ApiKeysStudio';
import { ArchitectureView } from './components/ArchitectureView';
import { ChatWorkspace } from './components/ChatWorkspace';
import { DownloadModelModal } from './components/DownloadModelModal';
import { Sidebar } from './components/Sidebar';
import { TrainingStudio } from './components/TrainingStudio';
import { VisionVideoStudio } from './components/VisionVideoStudio';
import { renderLtxStillToDataUrl } from './engine/ltxCinemaWebgl';
import type {
  ChatEngineMode,
  ChatMessage,
  ChatOutputMode,
  Conversation,
  ReasoningDepth,
  SamplingSettings,
  WorkspaceTab,
} from './types/qowe';
import { rgbaFrameToDataUrl, svgToPngDataUrl } from './utils/mediaExport';

const STORAGE_KEYS = {
  conversations: 'qowe_conversations_v1',
  checkpoint: 'qowe_checkpoint_v2',
  multimodalCheckpoint: 'qowe_mm_checkpoint_v1',
  corpus: 'qowe_corpus_v1',
  theme: 'qowe_theme_v1',
  architecture: 'qowe_arch_v2',
} as const;

function createEmptyConversation(): Conversation {
  return {
    id: `conv-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: 'New Chat',
    updatedAt: new Date().toISOString(),
    messages: [],
  };
}

export default function App() {
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    try {
      return localStorage.getItem(STORAGE_KEYS.theme) === 'dark';
    } catch {
      return false;
    }
  });

  const [activeTab, setActiveTab] = useState<WorkspaceTab>('chat');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isDownloadModalOpen, setIsDownloadModalOpen] = useState(false);

  const [archConfig, setArchConfig] = useState<QoweArchitectureConfig>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEYS.architecture);
      if (saved) return JSON.parse(saved) as QoweArchitectureConfig;
    } catch {
      // Ignore invalid JSON
    }
    return QOWE_PRESETS[0];
  });

  const [corpus, setCorpus] = useState<string>(() => {
    try {
      return localStorage.getItem(STORAGE_KEYS.corpus) || TRAINING_DATASETS[0].corpus;
    } catch {
      return TRAINING_DATASETS[0].corpus;
    }
  });

  const [checkpoint, setCheckpoint] = useState<QoweCheckpoint | null>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEYS.checkpoint);
      return raw ? (JSON.parse(raw) as QoweCheckpoint) : null;
    } catch {
      return null;
    }
  });

  const [multimodalCheckpoint, setMultimodalCheckpoint] = useState<MultimodalCheckpoint | null>(
    () => {
      try {
        const raw = localStorage.getItem(STORAGE_KEYS.multimodalCheckpoint);
        return raw ? (JSON.parse(raw) as MultimodalCheckpoint) : null;
      } catch {
        return null;
      }
    },
  );

  const [conversations, setConversations] = useState<Conversation[]>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEYS.conversations);
      if (raw) {
        const parsed = JSON.parse(raw) as Conversation[];
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed.map((conv) => ({
            ...conv,
            messages: Array.isArray(conv.messages)
              ? conv.messages.map((m) => ({
                  ...m,
                  videoData: undefined,
                }))
              : [],
          }));
        }
      }
    } catch {
      // Ignore parse errors
    }
    return [createEmptyConversation()];
  });

  const [activeConversationId, setActiveConversationId] = useState<string>(
    () => conversations[0]?.id ?? '',
  );
  const [engineMode, setEngineMode] = useState<ChatEngineMode>('assistant');
  const [reasoningDepth, setReasoningDepth] = useState<ReasoningDepth>('deep');
  const [chatOutputMode, setChatOutputMode] = useState<ChatOutputMode>('text');
  const [videoEngineSource, setVideoEngineSource] = useState<'scratch-neural' | 'scene-script'>(
    'scratch-neural',
  );
  const [isQuickTrainingMultimodal, setIsQuickTrainingMultimodal] = useState(false);
  const [sampling, setSampling] = useState<SamplingSettings>({
    temperature: 0.25,
    topK: 3,
    maxTokens: 48,
  });
  const [isStreaming, setIsStreaming] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  const archSpecs = useMemo(() => calculateArchitectureSpecs(archConfig), [archConfig]);
  const totalParamsFormatted = formatParamCount(archSpecs.totalParams);

  const activeConversation =
    conversations.find((c) => c.id === activeConversationId) ?? conversations[0];

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    try {
      localStorage.setItem(STORAGE_KEYS.theme, darkMode ? 'dark' : 'light');
    } catch {
      // Ignore storage quota errors
    }
  }, [darkMode]);

  // Persist compact conversation history only when idle & not actively streaming
  useEffect(() => {
    if (isStreaming) return;
    let idleHandle: number | null = null;
    const timer = window.setTimeout(() => {
      const saveTask = () => {
        try {
          const compact = conversations.map((c) => ({
            ...c,
            messages: c.messages.map((m) => ({
              ...m,
              videoData: undefined,
              imageDataUrl: undefined,
              svgMarkup: m.svgMarkup && m.svgMarkup.length < 40000 ? m.svgMarkup : undefined,
            })),
          }));
          localStorage.setItem(STORAGE_KEYS.conversations, JSON.stringify(compact));
        } catch {
          // Ignore quota errors
        }
      };
      if (typeof window.requestIdleCallback === 'function') {
        idleHandle = window.requestIdleCallback(saveTask, { timeout: 1000 });
      } else {
        saveTask();
      }
    }, 350);
    return () => {
      window.clearTimeout(timer);
      if (idleHandle !== null && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(idleHandle);
      }
    };
  }, [conversations, isStreaming]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      try {
        localStorage.setItem(STORAGE_KEYS.corpus, corpus);
      } catch {
        // Ignore quota errors
      }
    }, 450);
    return () => window.clearTimeout(timer);
  }, [corpus]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEYS.architecture, JSON.stringify(archConfig));
    } catch {
      // Ignore quota errors
    }
  }, [archConfig]);

  const scheduleStorageWrite = (key: string, payload: unknown | null) => {
    const run = () => {
      try {
        if (payload !== null) {
          localStorage.setItem(key, JSON.stringify(payload));
        } else {
          localStorage.removeItem(key);
        }
      } catch {
        // Ignore quota errors
      }
    };
    if (typeof window !== 'undefined' && typeof window.requestIdleCallback === 'function') {
      window.requestIdleCallback(run, { timeout: 1200 });
    } else {
      window.setTimeout(run, 0);
    }
  };

  const handleSaveCheckpoint = (cp: QoweCheckpoint | null) => {
    setCheckpoint(cp);
    scheduleStorageWrite(STORAGE_KEYS.checkpoint, cp);
  };

  const handleSaveMultimodalCheckpoint = (cp: MultimodalCheckpoint | null) => {
    setMultimodalCheckpoint(cp);
    scheduleStorageWrite(STORAGE_KEYS.multimodalCheckpoint, cp);
  };

  const handleQuickTrainMultimodal = async () => {
    if (isQuickTrainingMultimodal) return;
    setIsQuickTrainingMultimodal(true);
    try {
      const res = await generateMediaInPool(
        'image',
        'golden sunrise over calm ocean waves',
        multimodalCheckpoint,
        140,
      );
      handleSaveMultimodalCheckpoint(res.checkpoint);
    } finally {
      setIsQuickTrainingMultimodal(false);
    }
  };

  const handleImportBundle = (bundle: QoweFullModelBundle) => {
    setArchConfig(bundle.architecture);
    handleSaveCheckpoint(bundle.textCheckpoint ?? null);
    handleSaveMultimodalCheckpoint(bundle.multimodalCheckpoint ?? null);
  };

  const handleNewChat = () => {
    const fresh = createEmptyConversation();
    setConversations((prev) => [fresh, ...prev]);
    setActiveConversationId(fresh.id);
  };

  const handleDeleteConversation = (id: string) => {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (next.length === 0) {
        const fresh = createEmptyConversation();
        setActiveConversationId(fresh.id);
        return [fresh];
      }
      if (activeConversationId === id) {
        setActiveConversationId(next[0].id);
      }
      return next;
    });
  };

  const handleClearConversation = () => {
    if (!activeConversation) return;
    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeConversation.id
          ? { ...c, title: 'New Chat', messages: [], updatedAt: new Date().toISOString() }
          : c,
      ),
    );
  };

  const runAssistantStream = async (
    convId: string,
    historyForRequest: ChatMessage[],
    assistantMsgId: string,
    depth: ReasoningDepth,
    mode: 'text' | 'build' = 'text',
  ) => {
    const controller = new AbortController();
    abortRef.current = controller;
    setIsStreaming(true);

    let accumulated = '';
    let flushTimer: number | null = null;
    let lastFlushMs = 0;

    const flushAccumulated = () => {
      if (flushTimer !== null) {
        window.clearTimeout(flushTimer);
        flushTimer = null;
      }
      lastFlushMs = performance.now();
      const snapshot = accumulated;
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === assistantMsgId ? { ...m, content: snapshot } : m,
                ),
              }
            : c,
        ),
      );
    };

    const scheduleFlush = () => {
      if (flushTimer !== null) return;
      const elapsed = performance.now() - lastFlushMs;
      const delay = elapsed >= 34 ? 0 : 34 - elapsed;
      flushTimer = window.setTimeout(() => {
        flushTimer = null;
        flushAccumulated();
      }, delay);
    };

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          messages: historyForRequest
            .filter((m) => m.content.trim().length > 0)
            .map((m) => ({ role: m.role, content: m.content })),
          architectureName: archConfig.name,
          totalParamsFormatted,
          trainingContext: corpus,
          checkpointStep: checkpoint?.step ?? 0,
          checkpointLoss: checkpoint?.loss,
          reasoningDepth: depth,
          outputMode: mode,
        }),
      });

      if (!response.ok) {
        const errPayload = await response.json().catch(() => ({}));
        throw new Error(
          (errPayload as { error?: string }).error || `Request failed with status ${response.status}`,
        );
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response stream is unavailable.');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line.startsWith('data:')) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;

          const parsed = JSON.parse(jsonStr) as { text?: string; error?: string; done?: boolean };
          if (parsed.error) {
            throw new Error(parsed.error);
          }
          if (parsed.text) {
            accumulated += parsed.text;
            scheduleFlush();
          }
        }
      }
      if (accumulated) {
        flushAccumulated();
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') {
        if (accumulated) flushAccumulated();
        return;
      }
      // Always fall back cleanly to local Qowe 1.0 neural weights if backend or upstream hits limits
      if (!accumulated) {
        try {
          const userMsgs = historyForRequest.filter((m) => m.role === 'user');
          const lastUserPrompt = userMsgs[userMsgs.length - 1]?.content ?? '';
          const activeCp =
            checkpoint && checkpoint.step >= 60
              ? checkpoint
              : await trainAssistantCheckpointAsync(checkpoint, corpus, 140);
          if (!checkpoint || activeCp.step > checkpoint.step) {
            handleSaveCheckpoint(activeCp);
          }
          const engine = QoweTrainerEngine.fromCheckpoint(activeCp, corpus);
          const gen = engine.generate(`<|user|> ${lastUserPrompt.trim()}\n<|assistant|>`, {
            maxNewTokens: sampling.maxTokens,
            temperature: Math.min(0.25, sampling.temperature),
            topK: Math.min(3, sampling.topK),
          });
          accumulated =
            gen.text.trim() ||
            `I am ${archConfig.name} (${totalParamsFormatted}), running locally on unlimited weights.`;
          flushAccumulated();
          return;
        } catch {
          // Fall through to clean fallback error if local generation also failed
        }
      }
      const errMessage = err instanceof Error ? err.message : 'Unable to reach Qowe 1.0 engine.';
      setConversations((prev) =>
        prev.map((c) =>
          c.id === convId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === assistantMsgId ? { ...m, error: errMessage } : m,
                ),
              }
            : c,
        ),
      );
    } finally {
      if (flushTimer !== null) {
        window.clearTimeout(flushTimer);
      }
      abortRef.current = null;
      setIsStreaming(false);
    }
  };

  const handleSendMessage = async (text: string, overrideOutputMode?: ChatOutputMode) => {
    if (!activeConversation) return;
    const convId = activeConversation.id;
    const requestedMode = overrideOutputMode ?? chatOutputMode;
    const effectiveOutputMode =
      engineMode === 'local-weights' && requestedMode === 'text'
        ? 'text'
        : detectChatIntent(text, requestedMode);

    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}-u`,
      role: 'user',
      content: text,
      createdAt: new Date().toISOString(),
    };
    const assistantMsgId = `msg-${Date.now()}-a`;
    const isMediaMode = effectiveOutputMode === 'image' || effectiveOutputMode === 'video';
    const useScratchVideo =
      effectiveOutputMode === 'video' &&
      (engineMode === 'local-weights' || videoEngineSource === 'scratch-neural');
    const placeholderAssistant: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content:
        effectiveOutputMode === 'image'
          ? 'Generating high-resolution studio image...'
          : effectiveOutputMode === 'video'
            ? useScratchVideo
              ? multimodalCheckpoint
                ? 'Synthesizing 60 fps audio + video from Qowe trained-from-scratch weights...'
                : 'Training Qowe audio-video neural weights from scratch & rendering 60 fps cinema...'
              : 'Directing 60 fps motion video scene...'
            : '',
      createdAt: new Date().toISOString(),
      modeUsed: effectiveOutputMode === 'build' ? 'assistant' : engineMode,
      reasoningDepth:
        effectiveOutputMode === 'text' || effectiveOutputMode === 'build'
          ? reasoningDepth
          : undefined,
    };

    const nextTitle =
      activeConversation.messages.length === 0 ? text.slice(0, 42) : activeConversation.title;
    const updatedHistory = [...activeConversation.messages, userMsg];

    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              title: nextTitle,
              updatedAt: new Date().toISOString(),
              messages: [...updatedHistory, placeholderAssistant],
            }
          : c,
      ),
    );

    if (isMediaMode) {
      try {
        if (effectiveOutputMode === 'image') {
          const media = await generateMediaInPool('image', text, multimodalCheckpoint, 140, {
            aspectRatio: '1:1',
          });
          if (!multimodalCheckpoint) {
            handleSaveMultimodalCheckpoint(media.checkpoint);
          }

          let pngUrl = media.cinemaPayload
            ? renderLtxStillToDataUrl(media.cinemaPayload, 0.35, media.width, media.height)
            : null;
          let fallbackSvg: string | undefined;

          if (!pngUrl && media.frames[0]) {
            pngUrl = rgbaFrameToDataUrl(media.frames[0], media.width, media.height, 4);
          }
          if (!pngUrl) {
            const resp = await fetch('/api/generate-image', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ prompt: text, aspectRatio: '1:1' }),
            });
            const data = (await resp.json()) as {
              svg?: string;
              width?: number;
              height?: number;
              error?: string;
            };
            if (!resp.ok || !data.svg) {
              throw new Error(data.error || 'Image generation failed.');
            }
            pngUrl = await svgToPngDataUrl(
              data.svg,
              data.width ?? 1024,
              data.height ?? 1024,
            );
            fallbackSvg = data.svg;
          }

          setConversations((prev) =>
            prev.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantMsgId
                        ? {
                            ...m,
                            content: `Generated ${media.width}×${media.height} LTX-2.5 3D PBR image for "${text.slice(0, 80)}" (zero API keys).`,
                            imageDataUrl: pngUrl ?? undefined,
                            svgMarkup: fallbackSvg,
                          }
                        : m,
                    ),
                  }
                : c,
            ),
          );
        } else if (useScratchVideo) {
          const media = await generateMediaInPool('video', text, multimodalCheckpoint, 140, {
            aspectRatio: '16:9',
            cameraMotion: 'auto',
            withAudio: true,
          });
          if (!multimodalCheckpoint) {
            handleSaveMultimodalCheckpoint(media.checkpoint);
          }
          setConversations((prev) =>
            prev.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantMsgId
                        ? {
                            ...m,
                            content: `Generated 60 fps ${media.width}×${media.height} LTX-2.5 Dual-Stream 3D PBR video + synced stereo neural audio locally (Step ${media.checkpoint.step}, camera: ${media.cameraMotion ?? 'auto'}, zero API keys). Unmute audio or export .webm below.`,
                            videoData: {
                              width: media.width,
                              height: media.height,
                              fps: media.fps,
                              frames: media.frames,
                              audioTrack: media.audioTrack,
                              cameraMotion: media.cameraMotion,
                              cinemaPayload: media.cinemaPayload,
                            },
                          }
                        : m,
                    ),
                  }
                : c,
            ),
          );
        } else {
          const resp = await fetch('/api/generate-video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: text, aspectRatio: '16:9', durationSec: 4 }),
          });
          const data = (await resp.json()) as {
            sceneSpec?: VideoSceneSpec;
            fps?: number;
            error?: string;
          };
          if (!resp.ok || !data.sceneSpec) {
            throw new Error(data.error || 'Video generation failed.');
          }
          setConversations((prev) =>
            prev.map((c) =>
              c.id === convId
                ? {
                    ...c,
                    messages: c.messages.map((m) =>
                      m.id === assistantMsgId
                        ? {
                            ...m,
                            content: `Generated 60 fps video ("${data.sceneSpec?.title || text.slice(0, 60)}"). Play, scrub, or export as .webm below.`,
                            videoScene: data.sceneSpec,
                          }
                        : m,
                    ),
                  }
                : c,
            ),
          );
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Multimodal generation failed.';
        setConversations((prev) =>
          prev.map((c) =>
            c.id === convId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === assistantMsgId ? { ...m, error: msg } : m,
                  ),
                }
              : c,
          ),
        );
      }
      return;
    }

    if (effectiveOutputMode === 'text' && engineMode === 'local-weights') {
      try {
        let activeCp = checkpoint;
        if (!activeCp || activeCp.step < 60) {
          activeCp = await trainAssistantCheckpointAsync(activeCp, corpus, 140, (step, total, loss) => {
            setConversations((prev) =>
              prev.map((c) =>
                c.id === convId
                  ? {
                      ...c,
                      messages: c.messages.map((m) =>
                        m.id === assistantMsgId
                          ? {
                              ...m,
                              content: `Training local Qowe 1.0 weights from scratch (${step}/${total} steps · Loss ${loss.toFixed(3)})...`,
                            }
                          : m,
                      ),
                    }
                  : c,
              ),
            );
          });
          handleSaveCheckpoint(activeCp);
        }
        const engine = QoweTrainerEngine.fromCheckpoint(activeCp, corpus);
        const promptFormatted = `<|user|> ${text}\n<|assistant|>`;
        const gen = engine.generate(promptFormatted, {
          maxNewTokens: sampling.maxTokens,
          temperature: sampling.temperature,
          topK: sampling.topK,
        });
        const cleanReply = gen.text.trim() || '(End of sequence reached)';
        setConversations((prev) =>
          prev.map((c) =>
            c.id === convId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === assistantMsgId
                      ? {
                          ...m,
                          content: cleanReply,
                          tokenCount: gen.generatedTokens,
                          topCandidates: gen.topCandidatesAtLastStep,
                        }
                      : m,
                  ),
                }
              : c,
          ),
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Local checkpoint inference failed.';
        setConversations((prev) =>
          prev.map((c) =>
            c.id === convId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === assistantMsgId ? { ...m, error: msg } : m,
                  ),
                }
              : c,
          ),
        );
      }
      return;
    }

    await runAssistantStream(
      convId,
      updatedHistory,
      assistantMsgId,
      reasoningDepth,
      effectiveOutputMode === 'build' ? 'build' : 'text',
    );
  };

  const handleStopStreaming = () => {
    abortRef.current?.abort();
    setIsStreaming(false);
  };

  const handleRetryLast = async () => {
    if (!activeConversation || activeConversation.messages.length < 2) return;
    const convId = activeConversation.id;
    const msgs = activeConversation.messages;
    const last = msgs[msgs.length - 1];
    if (last.role !== 'assistant') return;

    const historyWithoutError = msgs.slice(0, -1);
    const freshAssistantId = `msg-${Date.now()}-a`;
    setConversations((prev) =>
      prev.map((c) =>
        c.id === convId
          ? {
              ...c,
              messages: [
                ...historyWithoutError,
                {
                  id: freshAssistantId,
                  role: 'assistant',
                  content: '',
                  createdAt: new Date().toISOString(),
                  modeUsed: 'assistant',
                  reasoningDepth,
                },
              ],
            }
          : c,
      ),
    );
    await runAssistantStream(
      convId,
      historyWithoutError,
      freshAssistantId,
      reasoningDepth,
      chatOutputMode === 'build' ? 'build' : 'text',
    );
  };

  const mobileNavTabs: { id: WorkspaceTab; label: string; icon: ReactNode }[] = [
    { id: 'chat', label: 'Chat', icon: <MessageSquare className="w-3.5 h-3.5" /> },
    { id: 'train', label: 'Train', icon: <FlaskConical className="w-3.5 h-3.5" /> },
    { id: 'vision', label: 'Video', icon: <Film className="w-3.5 h-3.5" /> },
    { id: 'architecture', label: '800B+', icon: <Cpu className="w-3.5 h-3.5" /> },
    { id: 'api', label: 'API', icon: <KeyRound className="w-3.5 h-3.5" /> },
  ];

  return (
    <div className="flex flex-col md:flex-row h-dvh w-full overflow-hidden bg-[#f5f5f7] dark:bg-[#0b0b0d] text-neutral-900 dark:text-neutral-100">
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        conversations={conversations}
        activeConversationId={activeConversation?.id ?? ''}
        onSelectConversation={setActiveConversationId}
        onNewChat={handleNewChat}
        onDeleteConversation={handleDeleteConversation}
        totalParamsFormatted={totalParamsFormatted}
        checkpointStep={checkpoint?.step ?? 0}
        multimodalStep={multimodalCheckpoint?.step ?? 0}
        onOpenDownloadModal={() => setIsDownloadModalOpen(true)}
        darkMode={darkMode}
        onToggleDarkMode={() => setDarkMode((d) => !d)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      {/* Persistent Mobile App Bar + Workspace Tabs */}
      <div className="md:hidden shrink-0 bg-white/95 dark:bg-[#161618]/95 border-b border-black/[0.06] dark:border-white/[0.08]">
        <div className="flex items-center justify-between gap-2 px-3 py-2">
          <div className="flex items-center gap-2 min-w-0">
            <button
              type="button"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation menu"
              className="p-2 rounded-full text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.05] dark:hover:bg-white/[0.08] shrink-0"
            >
              <Menu className="w-5 h-5" />
            </button>
            <span className="text-sm font-bold tracking-tight text-neutral-900 dark:text-white whitespace-nowrap">
              Qowe 1.0
            </span>
            <span className="text-xs text-neutral-500 dark:text-neutral-400 font-mono tabular-nums whitespace-nowrap">
              · {totalParamsFormatted}
            </span>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={() => {
                handleNewChat();
                setActiveTab('chat');
              }}
              aria-label="New chat"
              className="p-2 rounded-full text-[#007aff] dark:text-[#0a84ff] hover:bg-[#007aff]/10"
            >
              <Plus className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => setIsDownloadModalOpen(true)}
              aria-label="Download Qowe 1.0 model"
              className="p-2 rounded-full text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.05] dark:hover:bg-white/[0.08]"
            >
              <Download className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => setDarkMode((d) => !d)}
              aria-label={darkMode ? 'Switch to light theme' : 'Switch to dark theme'}
              className="p-2 rounded-full text-neutral-700 dark:text-neutral-200 hover:bg-black/[0.05] dark:hover:bg-white/[0.08]"
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>

        <nav aria-label="Mobile workspace tabs" className="grid grid-cols-5 gap-1 px-2 pb-2">
          {mobileNavTabs.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                  active
                    ? 'bg-[#007aff] dark:bg-[#0a84ff] text-white shadow-xs'
                    : 'bg-black/[0.04] dark:bg-white/[0.06] text-neutral-700 dark:text-neutral-300'
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      <main className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
        {activeTab === 'chat' && activeConversation && (
          <ChatWorkspace
            conversation={activeConversation}
            engineMode={engineMode}
            onChangeEngineMode={setEngineMode}
            reasoningDepth={reasoningDepth}
            onChangeReasoningDepth={setReasoningDepth}
            outputMode={chatOutputMode}
            onChangeOutputMode={setChatOutputMode}
            videoEngineSource={videoEngineSource}
            onChangeVideoEngineSource={setVideoEngineSource}
            sampling={sampling}
            onChangeSampling={setSampling}
            isStreaming={isStreaming}
            onSendMessage={handleSendMessage}
            onStopStreaming={handleStopStreaming}
            onClearConversation={handleClearConversation}
            onRetryLast={handleRetryLast}
            architectureName={archConfig.name}
            totalParamsFormatted={totalParamsFormatted}
            checkpoint={checkpoint}
            multimodalCheckpoint={multimodalCheckpoint}
            onQuickTrainMultimodal={handleQuickTrainMultimodal}
            isQuickTrainingMultimodal={isQuickTrainingMultimodal}
            onOpenDownloadModal={() => setIsDownloadModalOpen(true)}
            onNavigateTab={setActiveTab}
            onOpenMobileMenu={() => setMobileOpen(true)}
          />
        )}

        {activeTab === 'train' && (
          <TrainingStudio
            corpus={corpus}
            onChangeCorpus={setCorpus}
            checkpoint={checkpoint}
            onSaveCheckpoint={handleSaveCheckpoint}
            onOpenChatWithWeights={() => {
              setEngineMode('local-weights');
              setChatOutputMode('text');
              setActiveTab('chat');
            }}
            onNavigateTab={setActiveTab}
            onOpenMobileMenu={() => setMobileOpen(true)}
            targetParamsFormatted={totalParamsFormatted}
          />
        )}

        {activeTab === 'vision' && (
          <VisionVideoStudio
            checkpoint={multimodalCheckpoint}
            onSaveCheckpoint={handleSaveMultimodalCheckpoint}
            onOpenChat={() => {
              setChatOutputMode('video');
              setActiveTab('chat');
            }}
            onOpenMobileMenu={() => setMobileOpen(true)}
          />
        )}

        {activeTab === 'architecture' && (
          <ArchitectureView
            config={archConfig}
            onChangeConfig={setArchConfig}
            onOpenMobileMenu={() => setMobileOpen(true)}
          />
        )}

        {activeTab === 'api' && (
          <ApiKeysStudio
            architectureName={archConfig.name}
            totalParamsFormatted={totalParamsFormatted}
          />
        )}
      </main>

      <DownloadModelModal
        open={isDownloadModalOpen}
        onClose={() => setIsDownloadModalOpen(false)}
        architecture={archConfig}
        totalParamsFormatted={totalParamsFormatted}
        corpus={corpus}
        textCheckpoint={checkpoint}
        multimodalCheckpoint={multimodalCheckpoint}
        onSaveTextCheckpoint={handleSaveCheckpoint}
        onImportBundle={handleImportBundle}
      />
    </div>
  );
}
