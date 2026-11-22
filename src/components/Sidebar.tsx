import React from 'react';
import {
  Cpu,
  Download,
  Film,
  FlaskConical,
  KeyRound,
  MessageSquare,
  Moon,
  Plus,
  Sun,
  Trash2,
  X,
} from 'lucide-react';
import type { Conversation, WorkspaceTab } from '../types/qowe';

interface SidebarProps {
  activeTab: WorkspaceTab;
  onSelectTab: (tab: WorkspaceTab) => void;
  conversations: Conversation[];
  activeConversationId: string;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: string) => void;
  totalParamsFormatted: string;
  checkpointStep: number;
  multimodalStep: number;
  onOpenDownloadModal: () => void;
  darkMode: boolean;
  onToggleDarkMode: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  totalParamsFormatted,
  checkpointStep,
  multimodalStep,
  onOpenDownloadModal,
  darkMode,
  onToggleDarkMode,
  mobileOpen,
  onCloseMobile,
}) => {
  const navItems: { id: WorkspaceTab; label: string; icon: React.ReactNode }[] = [
    { id: 'chat', label: 'Chat Workspace', icon: <MessageSquare className="w-4 h-4" /> },
    { id: 'train', label: 'Train LLM (Text)', icon: <FlaskConical className="w-4 h-4" /> },
    { id: 'vision', label: 'Image & Video (Scratch)', icon: <Film className="w-4 h-4" /> },
    { id: 'architecture', label: '800B+ Architecture', icon: <Cpu className="w-4 h-4" /> },
    { id: 'api', label: 'API Keys & URLs', icon: <KeyRound className="w-4 h-4" /> },
  ];

  const content = (
    <div className="flex flex-col h-full w-64 bg-white/95 dark:bg-[#161618]/95 border-r border-black/[0.06] dark:border-white/[0.08]">
      <div className="flex items-center justify-between px-4 py-4 border-b border-black/[0.06] dark:border-white/[0.08]">
        <span className="text-lg font-bold tracking-tight text-neutral-900 dark:text-white whitespace-nowrap shrink-0">
          Qowe 1.0
        </span>
        <button
          type="button"
          onClick={onCloseMobile}
          aria-label="Close navigation menu"
          className="md:hidden p-1.5 rounded-full text-neutral-500 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-white hover:bg-black/[0.05] dark:hover:bg-white/[0.08]"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="p-3">
        <button
          type="button"
          onClick={() => {
            onNewChat();
            onSelectTab('chat');
            onCloseMobile();
          }}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-full bg-[#007aff] hover:bg-[#0066d6] dark:bg-[#0a84ff] dark:hover:bg-[#1a8cff] text-white text-sm font-medium shadow-xs transition-colors whitespace-nowrap"
        >
          <Plus className="w-4 h-4" />
          <span>New Chat</span>
        </button>
      </div>

      <nav aria-label="Workspace sections" className="px-3 space-y-1">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                onSelectTab(item.id);
                onCloseMobile();
              }}
              className={`w-full flex items-center gap-2.5 px-3.5 py-2.5 rounded-2xl text-sm font-medium transition-colors whitespace-nowrap ${
                isActive
                  ? 'bg-black/[0.06] dark:bg-white/[0.1] text-neutral-900 dark:text-white'
                  : 'text-neutral-600 dark:text-neutral-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05] hover:text-neutral-900 dark:hover:text-white'
              }`}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-5 px-4 pb-2">
        <span className="text-xs font-medium text-neutral-500 dark:text-neutral-400">
          Recent Conversations
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-3 space-y-1">
        {conversations.map((conv) => {
          const isSelected = conv.id === activeConversationId && activeTab === 'chat';
          return (
            <div
              key={conv.id}
              className={`group flex items-center justify-between rounded-2xl transition-colors ${
                isSelected
                  ? 'bg-black/[0.06] dark:bg-white/[0.1] text-neutral-900 dark:text-white'
                  : 'text-neutral-600 dark:text-neutral-400 hover:bg-black/[0.03] dark:hover:bg-white/[0.05]'
              }`}
            >
              <button
                type="button"
                onClick={() => {
                  onSelectConversation(conv.id);
                  onSelectTab('chat');
                  onCloseMobile();
                }}
                className="flex-1 text-left px-3.5 py-2 text-sm truncate"
              >
                {conv.title || 'Untitled chat'}
              </button>
              {conversations.length > 1 && (
                <button
                  type="button"
                  onClick={() => onDeleteConversation(conv.id)}
                  aria-label={`Delete conversation ${conv.title || 'Untitled chat'}`}
                  className="p-1.5 mr-1.5 rounded-full text-neutral-400 hover:text-red-600 dark:hover:text-red-400 opacity-70 group-hover:opacity-100 transition-opacity"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          );
        })}
      </div>

      <div className="p-3 border-t border-black/[0.06] dark:border-white/[0.08] space-y-2">
        <div className="px-2 flex items-center gap-1.5 text-xs text-neutral-500 dark:text-neutral-400 tabular-nums whitespace-nowrap truncate">
          <span>{totalParamsFormatted}</span>
          <span aria-hidden="true">·</span>
          <span>{checkpointStep > 0 ? `LLM ${checkpointStep}` : 'LLM 0'}</span>
          <span aria-hidden="true">·</span>
          <span>{multimodalStep > 0 ? `MM ${multimodalStep}` : 'MM 0'}</span>
        </div>

        <button
          type="button"
          onClick={() => {
            onOpenDownloadModal();
            onCloseMobile();
          }}
          className="w-full flex items-center justify-between px-4 py-2.5 rounded-full bg-neutral-900 hover:bg-neutral-800 dark:bg-white dark:hover:bg-neutral-200 text-white dark:text-neutral-900 text-xs font-medium transition-colors whitespace-nowrap"
        >
          <span>Download Model</span>
          <Download className="w-3.5 h-3.5" />
        </button>

        <button
          type="button"
          onClick={onToggleDarkMode}
          aria-label={darkMode ? 'Switch to light theme' : 'Switch to dark theme'}
          className="w-full flex items-center justify-between px-4 py-2 rounded-full border border-black/[0.08] dark:border-white/[0.1] text-xs font-medium text-neutral-700 dark:text-neutral-300 hover:bg-black/[0.04] dark:hover:bg-white/[0.06] transition-colors"
        >
          <span>{darkMode ? 'Dark Theme' : 'Light Theme'}</span>
          {darkMode ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop fixed sidebar */}
      <aside className="hidden md:block shrink-0 h-full">{content}</aside>

      {/* Mobile slide-over drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 md:hidden flex">
          <button
            type="button"
            aria-label="Close menu overlay"
            onClick={onCloseMobile}
            className="fixed inset-0 bg-black/40 backdrop-blur-xs"
          />
          <aside className="relative z-50 h-full">{content}</aside>
        </div>
      )}
    </>
  );
};
