import { useState, useCallback, useMemo, useRef } from 'react'
import {
  Plus,
  ChevronDown,
  ChevronRight,
  FileText,
  Upload,
  BookOpen,
  Hash,
  Quote,
} from 'lucide-react'
import { useAppStore, type Session } from '../../stores/appStore'
import { useDocumentStore } from '../../stores/documentStore'
import { api, apiUpload } from '../../lib/api'

/* ---------- date grouping helpers ---------- */

function daysBetween(a: Date, b: Date): number {
  const msPerDay = 86_400_000
  return Math.floor((b.getTime() - a.getTime()) / msPerDay)
}

function groupSessions(sessions: Session[]): Record<string, Session[]> {
  const now = new Date()
  const groups: Record<string, Session[]> = {}

  const sorted = [...sessions].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  )

  for (const s of sorted) {
    const d = new Date(s.created_at)
    const diff = daysBetween(d, now)
    let label: string
    if (diff < 1) label = 'Today'
    else if (diff < 2) label = 'Yesterday'
    else if (diff < 7) label = 'This Week'
    else label = 'Older'

    ;(groups[label] ??= []).push(s)
  }
  return groups
}

/* ---------- component ---------- */

export function Sidebar() {
  const [sourcesOpen, setSourcesOpen] = useState(true)
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const session = useAppStore((s) => s.session)
  const sessions = useAppStore((s) => s.sessions)
  const setSession = useAppStore((s) => s.setSession)
  const setSessions = useAppStore((s) => s.setSessions)
  const setMessages = useAppStore((s) => s.setMessages)

  const sources = useDocumentStore((s) => s.sources)
  const structure = useDocumentStore((s) => s.structure)

  const grouped = useMemo(() => groupSessions(sessions), [sessions])

  /* stats */
  const totalChunks = sources.reduce((n, f) => n + (f.chunks ?? 0), 0)

  /* new chat */
  const handleNewChat = useCallback(async () => {
    try {
      const newSession = await api<Session>('/api/sessions', {
        method: 'POST',
        body: JSON.stringify({ title: 'New Chat' }),
      })
      setSession(newSession)
      setSessions([newSession, ...sessions])
      setMessages([])
    } catch {
      /* silent */
    }
  }, [sessions, setSession, setSessions, setMessages])

  /* switch session */
  const handleSelectSession = useCallback(
    async (s: Session) => {
      setSession(s)
      try {
        const res = await api<{ session: any; messages: any[] }>(`/api/sessions/${s.id}`)
        setMessages(res.messages ?? [])
      } catch {
        setMessages([])
      }
    },
    [setSession, setMessages]
  )

  /* PDF upload */
  const uploadFiles = useCallback(async (fileList: FileList | File[]) => {
    const fd = new FormData()
    for (const f of Array.from(fileList)) {
      fd.append('files', f)
    }
    try {
      await apiUpload('/api/ingest', fd)
      const docs = await api<any[]>('/api/sources/detailed')
      useDocumentStore.getState().setSources(Array.isArray(docs) ? docs : [])
    } catch {
      /* silent */
    }
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragging(false)
      if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files)
    },
    [uploadFiles]
  )

  return (
    <div className="flex h-full flex-col bg-bg-sidebar">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <span className="text-sm font-semibold text-text-primary tracking-tight">
          MBA Agent
        </span>
        <button
          onClick={handleNewChat}
          title="New Chat"
          className="flex items-center justify-center w-7 h-7 rounded-md text-text-secondary hover:bg-bg-sidebar-hover hover:text-text-primary transition-colors"
        >
          <Plus size={16} strokeWidth={2} />
        </button>
      </div>

      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto">
        {/* Progress card */}
        {structure && (
          <div className="mx-3 mt-3 rounded-lg bg-bg-card border border-border-light p-3">
            <div className="flex items-center justify-between text-xs text-text-secondary mb-1.5">
              <span className="font-medium">Progress</span>
              <span>
                {structure.progress.sections_complete}/{structure.progress.sections_total} sections
              </span>
            </div>
            <div className="h-1.5 rounded-full bg-border-light overflow-hidden">
              <div
                className="h-full rounded-full bg-accent transition-all duration-300"
                style={{ width: `${Math.min(structure.progress.pct, 100)}%` }}
              />
            </div>
            <div className="text-[11px] text-text-muted mt-1 text-right">
              {structure.progress.pct}%
            </div>
          </div>
        )}

        {/* Sources section */}
        <div className="mt-3">
          <button
            onClick={() => setSourcesOpen(!sourcesOpen)}
            className="flex w-full items-center gap-1.5 px-4 py-1.5 text-xs font-medium text-text-secondary uppercase tracking-wider hover:text-text-primary transition-colors"
          >
            {sourcesOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            Sources
            <span className="ml-auto text-text-muted font-normal normal-case">
              {sources.length}
            </span>
          </button>

          {sourcesOpen && (
            <div className="px-3 pb-2">
              {sources.map((f) => (
                <div
                  key={f.file}
                  className="flex items-center gap-2 px-2 py-1.5 rounded-md text-xs text-text-secondary hover:bg-bg-sidebar-hover transition-colors cursor-default"
                  title={f.file}
                >
                  <FileText size={13} className="shrink-0 text-text-muted" />
                  <span className="truncate flex-1">{f.label || f.file}</span>
                  {f.chunks > 0 && (
                    <span className="text-[10px] text-text-muted shrink-0">
                      {f.chunks}
                    </span>
                  )}
                </div>
              ))}

              {/* Drop zone */}
              <div
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragging(true)
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`
                  mt-2 flex flex-col items-center justify-center gap-1
                  rounded-lg border-2 border-dashed py-4 cursor-pointer
                  text-xs text-text-muted transition-colors
                  ${dragging ? 'border-accent bg-accent/5' : 'border-border-light hover:border-border'}
                `}
              >
                <Upload size={16} />
                <span>Drop PDFs here</span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.length) uploadFiles(e.target.files)
                  }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Sessions */}
        <div className="mt-2">
          <div className="px-4 py-1.5 text-xs font-medium text-text-secondary uppercase tracking-wider">
            Sessions
          </div>

          {Object.entries(grouped).map(([label, items]) => (
            <div key={label} className="mb-1">
              <div className="px-4 py-1 text-[10px] font-medium text-text-muted uppercase tracking-widest">
                {label}
              </div>
              {items.map((s) => (
                <button
                  key={s.id}
                  onClick={() => handleSelectSession(s)}
                  className={`
                    flex w-full items-center gap-2 px-4 py-1.5
                    text-xs text-left transition-colors rounded-none
                    ${
                      session?.id === s.id
                        ? 'bg-bg-sidebar-active text-text-primary font-medium'
                        : 'text-text-secondary hover:bg-bg-sidebar-hover'
                    }
                  `}
                >
                  <span className="truncate">{s.title}</span>
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Footer stats */}
      <div className="flex items-center gap-3 px-4 py-2.5 border-t border-border text-[10px] text-text-muted">
        <span className="flex items-center gap-1">
          <BookOpen size={10} />
          {sources.length}
        </span>
        <span className="flex items-center gap-1">
          <Hash size={10} />
          {totalChunks}
        </span>
        <span className="flex items-center gap-1">
          <Quote size={10} />
          {structure?.progress.total_words?.toLocaleString() ?? 0}w
        </span>
      </div>
    </div>
  )
}
