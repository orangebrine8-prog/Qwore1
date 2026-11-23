import React, { useEffect, useRef, useState } from 'react';
import { Check, Code2, Download, Globe, Loader2, Package, Upload, X } from 'lucide-react';
import type { QoweArchitectureConfig } from '../engine/architecture';
import {
  buildFullModelBundle,
  buildPytorchScript,
  buildStandaloneChatHtml,
  isTextCheckpointReadyForChat,
  QoweFullModelBundle,
  trainAssistantCheckpointAsync,
} from '../engine/exporter';
import type { MultimodalCheckpoint } from '../engine/multimodal';
import type { QoweCheckpoint } from '../engine/trainer';
import { triggerBlobDownload } from '../utils/mediaExport';

interface DownloadModelModalProps {
  open: boolean;
  onClose: () => void;
  architecture: QoweArchitectureConfig;
  totalParamsFormatted: string;
  corpus: string;
  textCheckpoint: QoweCheckpoint | null;
  multimodalCheckpoint: MultimodalCheckpoint | null;
  onSaveTextCheckpoint: (cp: QoweCheckpoint) => void;
  onImportBundle: (bundle: QoweFullModelBundle) => void;
}

export const DownloadModelModal: React.FC<DownloadModelModalProps> = ({
  open,
  onClose,
  architecture,
  totalParamsFormatted,
  corpus,
  textCheckpoint,
  multimodalCheckpoint,
  onSaveTextCheckpoint,
  onImportBundle,
}) => {
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isPreparing, setIsPreparing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) return;
    setStatusMsg(null);
    setErrorMsg(null);
    setIsPreparing(false);
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const isTextReady = isTextCheckpointReadyForChat(textCheckpoint);

  const prepareConvergedBundle = async (): Promise<QoweFullModelBundle> => {
    let activeTextCp = textCheckpoint;
    if (!isTextCheckpointReadyForChat(activeTextCp)) {
      setIsPreparing(true);
      setErrorMsg(null);
      setStatusMsg('Training assistant weights from scratch so exported model gives coherent replies...');
      activeTextCp = await trainAssistantCheckpointAsync(
        activeTextCp,
        corpus,
        140,
        (step, total, loss) => {
          setStatusMsg(`Training assistant weights (${step}/${total} steps · Loss ${loss.toFixed(3)})...`);
        },
      );
      onSaveTextCheckpoint(activeTextCp);
    }
    return buildFullModelBundle(architecture, activeTextCp, multimodalCheckpoint);
  };

  const handleDownloadStandaloneHtml = async () => {
    if (isPreparing) return;
    try {
      const bundle = await prepareConvergedBundle();
      const html = buildStandaloneChatHtml(bundle);
      const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
      triggerBlobDownload(blob, 'qowe-1.0-chat.html');
      setErrorMsg(null);
      setStatusMsg(
        `Downloaded standalone HTML chat app (Step ${bundle.textCheckpoint?.step ?? 0} · Loss ${bundle.textCheckpoint?.loss.toFixed(3) ?? 'N/A'}). Open it directly in any browser!`,
      );
    } catch (err) {
      setStatusMsg(null);
      setErrorMsg(err instanceof Error ? err.message : 'Unable to export standalone HTML chat.');
    } finally {
      setIsPreparing(false);
    }
  };

  const handleDownloadJsonBundle = async () => {
    if (isPreparing) return;
    try {
      const bundle = await prepareConvergedBundle();
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
      triggerBlobDownload(blob, 'qowe-1.0-model-bundle.json');
      setErrorMsg(null);
      setStatusMsg(
        `Downloaded Qowe 1.0 JSON bundle with trained assistant weights (Step ${bundle.textCheckpoint?.step ?? 0}) & inferenceSpec for external AI builders.`,
      );
    } catch (err) {
      setStatusMsg(null);
      setErrorMsg(err instanceof Error ? err.message : 'Unable to export model JSON bundle.');
    } finally {
      setIsPreparing(false);
    }
  };

  const handleDownloadPytorch = () => {
    const code = buildPytorchScript(architecture);
    const blob = new Blob([code], { type: 'text/x-python;charset=utf-8' });
    triggerBlobDownload(blob, 'qowe_1_0_model.py');
    setErrorMsg(null);
    setStatusMsg('Downloaded standalone PyTorch 100M+ multimodal model script.');
  };

  const handleUploadBundle = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result)) as QoweFullModelBundle;
        if (parsed.format !== 'qowe-1.0-bundle' || !parsed.architecture) {
          throw new Error('Selected file is not a valid Qowe 1.0 model bundle.');
        }
        onImportBundle(parsed);
        setErrorMsg(null);
        setStatusMsg('Imported Qowe 1.0 architecture and trained weights successfully.');
      } catch (err) {
        setStatusMsg(null);
        setErrorMsg(err instanceof Error ? err.message : 'Unable to parse model bundle file.');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="download-modal-title"
        className="w-full max-w-lg rounded-[28px] bg-white/95 dark:bg-[#1c1c1e]/95 border border-black/[0.08] dark:border-white/[0.1] shadow-2xl overflow-hidden"
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-black/[0.06] dark:border-white/[0.08]">
          <div>
            <h2
              id="download-modal-title"
              className="text-base font-semibold text-neutral-900 dark:text-white"
            >
              Download &amp; Export Qowe 1.0 Model
            </h2>
            <p className="text-xs text-neutral-500 dark:text-neutral-400 tabular-nums">
              {architecture.name} · {totalParamsFormatted} parameters
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close download modal"
            className="p-2 rounded-full text-neutral-500 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white hover:bg-black/[0.05] dark:hover:bg-white/[0.08]"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          {/* Checkpoint readiness summary */}
          <div className="rounded-2xl bg-[#f5f5f7] dark:bg-[#121214] p-4 space-y-2 text-xs tabular-nums">
            <div className="flex items-center justify-between gap-2">
              <span className="text-neutral-600 dark:text-neutral-400">Text Assistant Checkpoint</span>
              <span className="font-mono font-medium text-neutral-900 dark:text-neutral-100 text-right">
                {textCheckpoint
                  ? `Step ${textCheckpoint.step} · Loss ${textCheckpoint.loss.toFixed(3)}${isTextReady ? ' · Ready' : ' (will auto-converge)'}`
                  : 'Auto-trains assistant weights on export'}
              </span>
            </div>
            <div className="flex items-center justify-between gap-2">
              <span className="text-neutral-600 dark:text-neutral-400">Vision &amp; Video Checkpoint</span>
              <span className="font-mono font-medium text-neutral-900 dark:text-neutral-100 text-right">
                {multimodalCheckpoint
                  ? `Step ${multimodalCheckpoint.step} · MSE ${multimodalCheckpoint.loss.toFixed(4)}`
                  : 'Config ready (untrained)'}
              </span>
            </div>
          </div>

          <div className="space-y-2.5">
            <button
              type="button"
              onClick={handleDownloadStandaloneHtml}
              disabled={isPreparing}
              className="w-full flex items-center justify-between gap-3 p-4 rounded-2xl bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] disabled:opacity-60 text-white text-left transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                {isPreparing ? (
                  <Loader2 className="w-5 h-5 shrink-0 animate-spin" />
                ) : (
                  <Globe className="w-5 h-5 shrink-0" />
                )}
                <div className="min-w-0">
                  <div className="text-sm font-semibold">Download Standalone HTML Chat (.html)</div>
                  <div className="text-xs text-blue-100">
                    Self-contained browser chat page with trained Qowe 1.0 weights — open anywhere offline.
                  </div>
                </div>
              </div>
              <Download className="w-4 h-4 shrink-0" />
            </button>

            <button
              type="button"
              onClick={handleDownloadJsonBundle}
              disabled={isPreparing}
              className="w-full flex items-center justify-between gap-3 p-4 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] hover:bg-black/[0.04] dark:hover:bg-white/[0.06] disabled:opacity-60 text-left transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                <Package className="w-5 h-5 text-neutral-700 dark:text-neutral-300 shrink-0" />
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-neutral-900 dark:text-white">
                    Download Full Model Bundle (.json)
                  </div>
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">
                    Trained assistant weights + tokenizer + inferenceSpec JS for Claude or custom UIs.
                  </div>
                </div>
              </div>
              <Download className="w-4 h-4 text-neutral-500 shrink-0" />
            </button>

            <button
              type="button"
              onClick={handleDownloadPytorch}
              disabled={isPreparing}
              className="w-full flex items-center justify-between gap-3 p-4 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#f5f5f7] dark:bg-[#121214] hover:bg-black/[0.04] dark:hover:bg-white/[0.06] disabled:opacity-60 text-left transition-colors"
            >
              <div className="flex items-center gap-3 min-w-0">
                <Code2 className="w-5 h-5 text-neutral-700 dark:text-neutral-300 shrink-0" />
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-neutral-900 dark:text-white">
                    Download PyTorch 100M+ Script (.py)
                  </div>
                  <div className="text-xs text-neutral-500 dark:text-neutral-400">
                    Standalone runnable causal transformer + video latent head configured to {totalParamsFormatted}.
                  </div>
                </div>
              </div>
              <Download className="w-4 h-4 text-neutral-500 shrink-0" />
            </button>

            <div className="pt-1">
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleUploadBundle}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isPreparing}
                className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-full border border-dashed border-black/[0.15] dark:border-white/[0.18] text-xs font-medium text-neutral-700 dark:text-neutral-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.06] disabled:opacity-60 transition-colors"
              >
                <Upload className="w-4 h-4" />
                <span>Import Saved Model Bundle (.json)</span>
              </button>
            </div>
          </div>

          {statusMsg && (
            <div className="flex items-center gap-2 p-3.5 rounded-2xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900 text-xs text-emerald-800 dark:text-emerald-200">
              <Check className="w-4 h-4 shrink-0" />
              <span>{statusMsg}</span>
            </div>
          )}

          {errorMsg && (
            <div className="p-3.5 rounded-2xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 text-xs text-red-700 dark:text-red-300">
              {errorMsg}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
