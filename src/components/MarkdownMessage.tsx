import React, { useMemo, useState } from 'react';
import { Check, CheckSquare, Code2, Copy, Download, Play, Square } from 'lucide-react';
import { parseFullStackFiles } from '../engine/fullStackProject';
import { triggerDataUrlDownload } from '../utils/mediaExport';
import { FullStackProjectCard } from './FullStackProjectCard';

interface MarkdownMessageProps {
  content: string;
  isStreaming?: boolean;
}

function renderInlineFormatting(text: string): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const regex = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let keyIdx = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const token = match[0];
    if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code
          key={`code-${keyIdx++}`}
          className="px-1.5 py-0.5 mx-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 font-mono text-[0.875em]"
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={`bold-${keyIdx++}`} className="font-semibold text-slate-900 dark:text-white">
          {token.slice(2, -2)}
        </strong>,
      );
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return parts;
}

function buildSandboxSrcDoc(language: string, code: string): string {
  const lang = language.toLowerCase();
  if (lang === 'svg') {
    return `<!doctype html>
<html>
<head><meta charset="utf-8"><style>html,body{margin:0;height:100%;display:flex;align-items:center;justify-content:center;background:#0f172a;}svg{max-width:100%;max-height:100%;}</style></head>
<body>${code}</body>
</html>`;
  }
  return code;
}

const CodeBlock = React.memo(function CodeBlock({
  language,
  code,
  isStreaming = false,
}: {
  language: string;
  code: string;
  isStreaming?: boolean;
}) {
  const baseLang = (language.trim().split(/[\s:]/)[0] || '').toLowerCase();
  const isRunnable = baseLang === 'html' || baseLang === 'svg';
  const [manualViewMode, setManualViewMode] = useState<'preview' | 'code' | null>(null);
  const [copied, setCopied] = useState(false);

  const viewMode: 'preview' | 'code' =
    manualViewMode ?? (isRunnable && !isStreaming ? 'preview' : 'code');
  const setViewMode = (mode: 'preview' | 'code') => setManualViewMode(mode);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // Clipboard may be unavailable in restricted contexts
    }
  };

  const handleDownloadArtifact = () => {
    const ext = baseLang === 'svg' ? 'svg' : baseLang === 'html' ? 'html' : baseLang || 'txt';
    const mime =
      baseLang === 'svg'
        ? 'image/svg+xml;charset=utf-8'
        : baseLang === 'html'
          ? 'text/html;charset=utf-8'
          : 'text/plain;charset=utf-8';
    const dataUrl = `data:${mime},${encodeURIComponent(code)}`;
    triggerDataUrlDownload(dataUrl, `qowe-1.0-build.${ext}`);
  };

  return (
    <div className="my-3 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#121214] text-neutral-100 overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2 border-b border-white/[0.08] bg-[#1c1c1e] text-xs text-neutral-400">
        <div className="flex items-center gap-2">
          <span className="font-mono text-neutral-300">{language || 'text'}</span>
          {isRunnable && (
            <div
              role="group"
              aria-label="Build artifact view"
              className="flex items-center p-0.5 rounded-full bg-white/[0.08]"
            >
              <button
                type="button"
                onClick={() => setViewMode('preview')}
                className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                  viewMode === 'preview'
                    ? 'bg-[#007aff] text-white'
                    : 'text-neutral-300 hover:text-white'
                }`}
              >
                <Play className="w-3 h-3" />
                <span>Run Live</span>
              </button>
              <button
                type="button"
                onClick={() => setViewMode('code')}
                className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                  viewMode === 'code'
                    ? 'bg-white/[0.16] text-white'
                    : 'text-neutral-300 hover:text-white'
                }`}
              >
                <Code2 className="w-3 h-3" />
                <span>Source Code</span>
              </button>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          {isRunnable && (
            <button
              type="button"
              onClick={handleDownloadArtifact}
              aria-label={`Download .${baseLang === 'svg' ? 'svg' : 'html'} artifact`}
              className="inline-flex items-center gap-1 text-xs text-neutral-300 hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              <Download className="w-3.5 h-3.5" />
              <span>.{baseLang === 'svg' ? 'svg' : 'html'}</span>
            </button>
          )}
          <button
            type="button"
            onClick={handleCopy}
            aria-label="Copy code snippet"
            className="inline-flex items-center gap-1.5 text-xs text-neutral-300 hover:text-white transition-colors whitespace-nowrap shrink-0"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
      </div>

      {isRunnable && viewMode === 'preview' && !isStreaming ? (
        <div className="bg-white dark:bg-[#121214]">
          <iframe
            title={`Qowe 1.0 live ${baseLang} build preview`}
            sandbox="allow-scripts"
            srcDoc={buildSandboxSrcDoc(baseLang, code)}
            className="w-full h-[300px] sm:h-[400px] border-0 block bg-white"
          />
        </div>
      ) : (
        <pre className="p-3.5 overflow-x-auto text-xs leading-relaxed font-mono">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
});

export const MarkdownMessage: React.FC<MarkdownMessageProps> = React.memo(({ content, isStreaming = false }) => {
  const fullStackProject = useMemo(
    () => (isStreaming ? null : parseFullStackFiles(content)),
    [content, isStreaming],
  );

  const segments = useMemo(() => {
    const out: { type: 'text' | 'code' | 'fullstack'; language?: string; body: string }[] = [];
    const fenceRegex = /```([^\n]*)\n([\s\S]*?)```/g;
    let lastIdx = 0;
    let insertedFullStackCard = false;
    let match: RegExpExecArray | null;

    const strippedSet = fullStackProject?.strippedRanges ?? [];
    const isStrippedRange = (start: number, end: number) =>
      strippedSet.some((r) => r.start === start && r.end === end);

    while ((match = fenceRegex.exec(content)) !== null) {
      if (match.index > lastIdx) {
        out.push({ type: 'text', body: content.slice(lastIdx, match.index) });
      }
      if (fullStackProject && isStrippedRange(match.index, fenceRegex.lastIndex)) {
        if (!insertedFullStackCard) {
          out.push({ type: 'fullstack', body: '' });
          insertedFullStackCard = true;
        }
      } else {
        out.push({
          type: 'code',
          language: match[1],
          body: match[2].replace(/\n$/, ''),
        });
      }
      lastIdx = fenceRegex.lastIndex;
    }
    if (lastIdx < content.length) {
      out.push({ type: 'text', body: content.slice(lastIdx) });
    }
    return out;
  }, [content, fullStackProject]);

  return (
    <div className="space-y-2.5 text-[15px] leading-relaxed text-slate-800 dark:text-slate-200">
      {segments.map((seg, idx) => {
        if (seg.type === 'fullstack' && fullStackProject) {
          return <FullStackProjectCard key={`fs-${fullStackProject.id}-${idx}`} project={fullStackProject} />;
        }
        if (seg.type === 'code') {
          return (
            <CodeBlock
              key={idx}
              language={seg.language || ''}
              code={seg.body}
              isStreaming={isStreaming && idx === segments.length - 1}
            />
          );
        }

        const lines = seg.body.split('\n');
        const elements: React.ReactNode[] = [];
        let listItems: { ordered: boolean; text: string; checked?: boolean }[] = [];

        const flushList = (keyBase: string) => {
          if (listItems.length === 0) return;
          const isOrdered = listItems[0].ordered;
          const hasTasks = listItems.some((item) => item.checked !== undefined);
          const Tag = isOrdered ? 'ol' : 'ul';
          elements.push(
            <Tag
              key={`list-${keyBase}`}
              className={
                hasTasks
                  ? 'pl-1 space-y-1.5 list-none'
                  : `pl-5 space-y-1 ${isOrdered ? 'list-decimal' : 'list-disc'}`
              }
            >
              {listItems.map((item, liIdx) => (
                <li key={liIdx} className={item.checked !== undefined ? 'flex items-start gap-2' : undefined}>
                  {item.checked !== undefined && (
                    item.checked ? (
                      <CheckSquare className="w-4 h-4 mt-1 text-emerald-600 dark:text-emerald-400 shrink-0" />
                    ) : (
                      <Square className="w-4 h-4 mt-1 text-slate-400 shrink-0" />
                    )
                  )}
                  <span>{renderInlineFormatting(item.text)}</span>
                </li>
              ))}
            </Tag>,
          );
          listItems = [];
        };

        lines.forEach((rawLine, lineIdx) => {
          const line = rawLine.trimEnd();
          const trimmed = line.trim();

          if (!trimmed) {
            flushList(`${idx}-${lineIdx}`);
            return;
          }

          if (trimmed.startsWith('### ')) {
            flushList(`${idx}-${lineIdx}`);
            elements.push(
              <h4
                key={`h4-${idx}-${lineIdx}`}
                className="text-base font-semibold text-slate-900 dark:text-white pt-2"
              >
                {renderInlineFormatting(trimmed.slice(4))}
              </h4>,
            );
            return;
          }

          if (trimmed.startsWith('## ') || trimmed.startsWith('# ')) {
            flushList(`${idx}-${lineIdx}`);
            const headingText = trimmed.replace(/^#{1,3}\s+/, '');
            elements.push(
              <h3
                key={`h3-${idx}-${lineIdx}`}
                className="text-lg font-semibold text-slate-900 dark:text-white pt-2"
              >
                {renderInlineFormatting(headingText)}
              </h3>,
            );
            return;
          }

          const taskMatch = trimmed.match(/^[-*]\s+\[([ xX])\]\s+(.*)$/);
          if (taskMatch) {
            if (listItems.length > 0 && listItems[0].ordered) {
              flushList(`${idx}-${lineIdx}`);
            }
            listItems.push({
              ordered: false,
              checked: taskMatch[1].toLowerCase() === 'x',
              text: taskMatch[2],
            });
            return;
          }

          const ulMatch = trimmed.match(/^[-*]\s+(.*)$/);
          if (ulMatch) {
            if (listItems.length > 0 && listItems[0].ordered) {
              flushList(`${idx}-${lineIdx}`);
            }
            listItems.push({ ordered: false, text: ulMatch[1] });
            return;
          }

          const olMatch = trimmed.match(/^\d+\.\s+(.*)$/);
          if (olMatch) {
            if (listItems.length > 0 && !listItems[0].ordered) {
              flushList(`${idx}-${lineIdx}`);
            }
            listItems.push({ ordered: true, text: olMatch[1] });
            return;
          }

          flushList(`${idx}-${lineIdx}`);
          elements.push(
            <p key={`p-${idx}-${lineIdx}`} className="whitespace-pre-wrap">
              {renderInlineFormatting(trimmed)}
            </p>,
          );
        });

        flushList(`${idx}-end`);
        return <React.Fragment key={idx}>{elements}</React.Fragment>;
      })}
    </div>
  );
});
