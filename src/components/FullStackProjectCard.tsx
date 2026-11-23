import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Check,
  Code2,
  Copy,
  Database,
  Download,
  FileCode2,
  Play,
  RotateCcw,
} from 'lucide-react';
import {
  buildFullStackSandboxSrcDoc,
  createProjectZipBlob,
  executeVirtualApiRequest,
  type ParsedFullStackProject,
} from '../engine/fullStackProject';
import { triggerBlobDownload, triggerDataUrlDownload } from '../utils/mediaExport';

interface FullStackProjectCardProps {
  project: ParsedFullStackProject;
}

type CardTab = 'preview' | 'files' | 'database';

export const FullStackProjectCard: React.FC<FullStackProjectCardProps> = React.memo(({ project }) => {
  const storageKey = `qowe_fsdb_${project.id}`;
  const [activeTab, setActiveTab] = useState<CardTab>(project.entryHtmlFile ? 'preview' : 'files');
  const [selectedFilePath, setSelectedFilePath] = useState<string>(
    project.files[0]?.path ?? '',
  );
  const [copiedPath, setCopiedPath] = useState<string | null>(null);
  const [previewNonce, setPreviewNonce] = useState(0);

  const [tables, setTables] = useState<Record<string, Record<string, unknown>[]>>(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        const parsed = JSON.parse(saved) as Record<string, Record<string, unknown>[]>;
        if (parsed && typeof parsed === 'object') return parsed;
      }
    } catch {
      // Ignore invalid JSON
    }
    return project.initialTables;
  });

  const tablesRef = useRef(tables);
  tablesRef.current = tables;

  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const bridgeChannelId = useMemo(
    () => `qowe_bridge_${project.id}_${previewNonce}`,
    [project.id, previewNonce],
  );

  // Sync DB tables cleanly when project updates
  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        const parsed = JSON.parse(saved) as Record<string, Record<string, unknown>[]>;
        if (parsed && typeof parsed === 'object') {
          setTables(parsed);
          return;
        }
      }
    } catch {
      // Ignore
    }
    setTables(project.initialTables);
  }, [storageKey, project.initialTables]);

  const srcDoc = useMemo(() => {
    if (!project.entryHtmlFile) return '';
    return buildFullStackSandboxSrcDoc(project.entryHtmlFile.content, bridgeChannelId);
  }, [project.entryHtmlFile, bridgeChannelId]);

  useEffect(() => {
    const onMessage = (ev: MessageEvent) => {
      if (ev.source !== iframeRef.current?.contentWindow) return;
      const msg = ev.data as {
        channel?: string;
        kind?: string;
        reqId?: number;
        method?: string;
        url?: string;
        bodyText?: string;
      } | null;
      if (!msg || msg.channel !== bridgeChannelId || msg.kind !== 'api-req') return;

      const { response, nextTables, mutated } = executeVirtualApiRequest(
        tablesRef.current,
        {
          method: msg.method || 'GET',
          url: msg.url || '/api',
          bodyText: msg.bodyText,
        },
        project.expectsArrayCollections,
      );

      if (mutated) {
        tablesRef.current = nextTables;
        setTables(nextTables);
        try {
          localStorage.setItem(storageKey, JSON.stringify(nextTables));
        } catch {
          // Ignore storage quota errors
        }
      }

      ev.source?.postMessage(
        {
          channel: bridgeChannelId,
          kind: 'api-res',
          reqId: msg.reqId,
          status: response.status,
          body: response.body,
        },
        '*' as WindowPostMessageOptions,
      );
    };

    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [bridgeChannelId, project.expectsArrayCollections, storageKey]);

  const activeFile = useMemo(
    () => project.files.find((f) => f.path === selectedFilePath) ?? project.files[0] ?? null,
    [project.files, selectedFilePath],
  );

  const handleResetDb = useCallback(() => {
    setTables(project.initialTables);
    tablesRef.current = project.initialTables;
    try {
      localStorage.removeItem(storageKey);
    } catch {
      // Ignore
    }
    setPreviewNonce((n) => n + 1);
  }, [project.initialTables, storageKey]);

  const handleDownloadZip = useCallback(() => {
    const blob = createProjectZipBlob(project.files);
    const slug = project.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '') || 'qowe-fullstack-app';
    triggerBlobDownload(blob, `${slug}.zip`);
  }, [project.files, project.title]);

  const handleCopyActiveFile = useCallback(async () => {
    if (!activeFile) return;
    try {
      await navigator.clipboard.writeText(activeFile.content);
      setCopiedPath(activeFile.path);
      window.setTimeout(() => setCopiedPath(null), 1800);
    } catch {
      // Ignore clipboard failures
    }
  }, [activeFile]);

  const handleDownloadActiveFile = useCallback(() => {
    if (!activeFile) return;
    const baseName = activeFile.path.split('/').pop() || 'file.txt';
    const dataUrl = `data:text/plain;charset=utf-8,${encodeURIComponent(activeFile.content)}`;
    triggerDataUrlDownload(dataUrl, baseName);
  }, [activeFile]);

  const tableEntries = Object.entries(tables);

  return (
    <div className="my-3 rounded-2xl border border-black/[0.08] dark:border-white/[0.1] bg-[#121214] text-neutral-100 overflow-hidden shadow-sm">
      {/* Top IDE Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 border-b border-white/[0.08] bg-[#1c1c1e] text-xs">
        <div className="flex flex-wrap items-center gap-2 min-w-0">
          <span className="font-semibold text-white truncate max-w-[200px]" title={project.title}>
            {project.title}
          </span>
          <span className="font-mono text-[11px] text-neutral-400 whitespace-nowrap">
            {project.files.length} files
            {project.routes.length > 0 ? ` · ${project.routes.length} API routes` : ''}
          </span>

          <div
            role="group"
            aria-label="Full-stack workspace view"
            className="flex items-center p-0.5 rounded-full bg-white/[0.08]"
          >
            {project.entryHtmlFile && (
              <button
                type="button"
                onClick={() => setActiveTab('preview')}
                className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                  activeTab === 'preview'
                    ? 'bg-[#007aff] text-white'
                    : 'text-neutral-300 hover:text-white'
                }`}
              >
                <Play className="w-3 h-3" />
                <span>Live App</span>
              </button>
            )}
            <button
              type="button"
              onClick={() => setActiveTab('files')}
              className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                activeTab === 'files'
                  ? 'bg-white/[0.16] text-white'
                  : 'text-neutral-300 hover:text-white'
              }`}
            >
              <Code2 className="w-3 h-3" />
              <span>Files</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('database')}
              className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium transition-colors whitespace-nowrap ${
                activeTab === 'database'
                  ? 'bg-white/[0.16] text-white'
                  : 'text-neutral-300 hover:text-white'
              }`}
            >
              <Database className="w-3 h-3" />
              <span>Database &amp; API</span>
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={handleResetDb}
            title="Reset sandbox database to initial seed rows"
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full border border-white/[0.12] text-xs text-neutral-300 hover:text-white hover:bg-white/[0.06] whitespace-nowrap"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Reset DB</span>
          </button>

          <button
            type="button"
            onClick={handleDownloadZip}
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#0a84ff] hover:bg-[#1a8cff] text-white text-xs font-medium whitespace-nowrap"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download .zip</span>
          </button>
        </div>
      </div>

      {/* Live App View */}
      {activeTab === 'preview' && project.entryHtmlFile && (
        <div className="bg-white dark:bg-[#121214]">
          <iframe
            key={previewNonce}
            ref={iframeRef}
            title={`${project.title} full-stack preview`}
            sandbox="allow-scripts"
            srcDoc={srcDoc}
            className="w-full h-[380px] sm:h-[460px] border-0 block bg-white"
          />
        </div>
      )}

      {/* Multi-File Explorer View */}
      {activeTab === 'files' && (
        <div>
          <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-white/[0.08] bg-[#161618] overflow-x-auto">
            <div className="flex items-center gap-1">
              {project.files.map((f) => {
                const selected = activeFile?.path === f.path;
                return (
                  <button
                    key={f.path}
                    type="button"
                    onClick={() => setSelectedFilePath(f.path)}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg font-mono text-xs transition-colors whitespace-nowrap ${
                      selected
                        ? 'bg-[#0a84ff]/20 text-[#60a5fa] border border-[#0a84ff]/40'
                        : 'text-neutral-400 hover:text-neutral-200 hover:bg-white/[0.05]'
                    }`}
                  >
                    <FileCode2 className="w-3 h-3 shrink-0" />
                    <span>{f.path}</span>
                  </button>
                );
              })}
            </div>

            {activeFile && (
              <div className="flex items-center gap-2.5 shrink-0 pl-2">
                <button
                  type="button"
                  onClick={handleDownloadActiveFile}
                  className="inline-flex items-center gap-1 text-xs text-neutral-300 hover:text-white whitespace-nowrap"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Save</span>
                </button>
                <button
                  type="button"
                  onClick={handleCopyActiveFile}
                  className="inline-flex items-center gap-1 text-xs text-neutral-300 hover:text-white whitespace-nowrap"
                >
                  {copiedPath === activeFile.path ? (
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                  <span>{copiedPath === activeFile.path ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
            )}
          </div>

          {activeFile && (
            <pre className="p-3.5 overflow-x-auto max-h-[420px] overflow-y-auto text-xs leading-relaxed font-mono text-neutral-200">
              <code>{activeFile.content}</code>
            </pre>
          )}
        </div>
      )}

      {/* Database & API Explorer View */}
      {activeTab === 'database' && (
        <div className="p-4 space-y-4 max-h-[460px] overflow-y-auto text-xs">
          {project.routes.length > 0 && (
            <div className="space-y-1.5">
              <div className="font-semibold text-neutral-200">Backend API Endpoints</div>
              <div className="flex flex-wrap gap-1.5">
                {project.routes.map((r, i) => (
                  <span
                    key={`${r.method}-${r.path}-${i}`}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/[0.06] border border-white/[0.08] font-mono text-[11px]"
                  >
                    <span className="text-emerald-400 font-semibold">{r.method}</span>
                    <span className="text-neutral-200">{r.path}</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-3">
            <div className="font-semibold text-neutral-200">
              Sandbox Database Tables ({tableEntries.length})
            </div>
            {tableEntries.length === 0 ? (
              <p className="text-neutral-400">
                No SQL tables detected yet. Interact with the Live App to auto-create collections.
              </p>
            ) : (
              tableEntries.map(([tableName, rows]) => {
                const columns =
                  rows.length > 0 ? Object.keys(rows[0]).slice(0, 8) : [];
                return (
                  <div
                    key={tableName}
                    className="rounded-xl border border-white/[0.08] bg-[#18181b] overflow-hidden"
                  >
                    <div className="flex items-center justify-between px-3 py-2 bg-white/[0.04] border-b border-white/[0.06]">
                      <span className="font-mono font-semibold text-neutral-200">{tableName}</span>
                      <span className="font-mono text-neutral-400 tabular-nums">
                        {rows.length} {rows.length === 1 ? 'row' : 'rows'}
                      </span>
                    </div>
                    {rows.length > 0 ? (
                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse font-mono text-[11px]">
                          <thead>
                            <tr className="border-b border-white/[0.06] text-neutral-400">
                              {columns.map((col) => (
                                <th key={col} className="px-3 py-1.5 font-medium whitespace-nowrap">
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-white/[0.04]">
                            {rows.slice(0, 20).map((row, idx) => (
                              <tr key={idx} className="hover:bg-white/[0.03]">
                                {columns.map((col) => {
                                  const val = row[col];
                                  const display =
                                    val === null || val === undefined
                                      ? 'null'
                                      : typeof val === 'object'
                                        ? JSON.stringify(val)
                                        : String(val);
                                  return (
                                    <td
                                      key={col}
                                      className="px-3 py-1.5 text-neutral-200 max-w-[180px] truncate"
                                      title={display}
                                    >
                                      {display}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="px-3 py-2 text-neutral-500 font-mono">(empty table)</div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
});
