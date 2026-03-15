import {
  useState,
  useRef,
  useEffect,
  useCallback,
  type KeyboardEvent,
} from 'react'
import {
  CheckCircle2,
  Circle,
  CircleDot,
  ChevronDown,
  ChevronRight,
  Send,
  Loader2,
  Brain,
  StopCircle,
  RefreshCw,
} from 'lucide-react'
import { marked } from 'marked'
import { useAppStore } from '../../stores/appStore'
import { useDocumentStore, type PaperSection } from '../../stores/documentStore'
import { apiRaw, api } from '../../lib/api'
import { MODES, MODEL_OPTIONS, THINK_OPTIONS } from '../../lib/constants'
import { toast } from '../modals/Toast'

// Configure marked for safe, compact output
marked.setOptions({ breaks: true, gfm: true })

function renderMarkdown(text: string): string {
  return marked.parse(text, { async: false }) as string
}

/* ================================================================
   Structure tree (top half)
   ================================================================ */

function StatusIcon({ status }: { status: string }) {
  if (status === 'complete' || status === 'done') {
    return <CheckCircle2 size={14} className="text-green shrink-0" />
  }
  if (status === 'in_progress' || status === 'in-progress' || status === 'draft' || status === 'drafting') {
    return <CircleDot size={14} className="text-accent shrink-0" />
  }
  if (status === 'outline') {
    return <CircleDot size={14} className="text-text-muted shrink-0" />
  }
  return <Circle size={14} className="text-text-muted shrink-0" />
}

interface SectionNodeProps {
  section: PaperSection
  childSections: PaperSection[]
  depth: number
}

function SectionNode({ section, childSections, depth }: SectionNodeProps) {
  const [open, setOpen] = useState(true)
  const sectionId = useAppStore((s) => s.sectionId)
  const setSectionId = useAppStore((s) => s.setSectionId)
  const isActive = sectionId === section.id
  const hasChildren = childSections.length > 0
  const pct = section.target_words > 0
    ? Math.min(100, Math.round((section.word_count / section.target_words) * 100))
    : 0

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => setSectionId(section.id)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') setSectionId(section.id)
        }}
        className={`
          flex w-full items-center gap-1.5 rounded-md px-2 py-1 text-left text-xs transition-colors cursor-pointer
          ${isActive ? 'bg-bg-sidebar-active text-text-primary font-medium' : 'text-text-secondary hover:bg-bg-sidebar-hover'}
        `}
        style={{ paddingLeft: `${8 + depth * 14}px` }}
      >
        {hasChildren ? (
          <button
            onClick={(e) => { e.stopPropagation(); setOpen(!open) }}
            className="shrink-0"
          >
            {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          </button>
        ) : (
          <span className="w-3 shrink-0" />
        )}
        <StatusIcon status={section.status} />
        <span className="truncate flex-1">{section.title}</span>
        <span className="text-[10px] text-text-muted shrink-0 tabular-nums">
          {section.word_count}/{section.target_words}
        </span>
      </div>

      {isActive && section.target_words > 0 && (
        <div className="mx-2 mb-1" style={{ paddingLeft: `${22 + depth * 14}px` }}>
          <div className="h-1 rounded-full bg-border-light overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${pct >= 80 ? 'bg-green' : 'bg-accent'}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {open && hasChildren && childSections.map((child) => (
        <SectionNode
          key={child.id}
          section={child}
          childSections={
            useDocumentStore.getState().structure?.sections
              .filter((s) => s.parent_id === child.id)
              .sort((a, b) => a.order - b.order) ?? []
          }
          depth={depth + 1}
        />
      ))}
    </div>
  )
}

function StructureTree() {
  const structure = useDocumentStore((s) => s.structure)

  if (!structure) {
    return (
      <div className="flex items-center justify-center h-full text-xs text-text-muted p-4 text-center">
        No paper structure loaded.<br />Click ⚡ to create one.
      </div>
    )
  }

  const roots = structure.sections.filter((s) => !s.parent_id).sort((a, b) => a.order - b.order)

  return (
    <div className="p-2 overflow-y-auto">
      <div className="px-2 pb-1.5 text-[10px] font-medium text-text-muted uppercase tracking-widest">
        Structure
      </div>
      {roots.map((root) => (
        <SectionNode
          key={root.id}
          section={root}
          childSections={structure.sections.filter((s) => s.parent_id === root.id).sort((a, b) => a.order - b.order)}
          depth={0}
        />
      ))}
    </div>
  )
}

/* ================================================================
   Chat panel (bottom half)
   ================================================================ */

function ChatPanel() {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const [input, setInput] = useState('')
  const [showAllModes, setShowAllModes] = useState(false)

  const mode = useAppStore((s) => s.mode)
  const setMode = useAppStore((s) => s.setMode)
  const session = useAppStore((s) => s.session)
  const messages = useAppStore((s) => s.messages)
  const loading = useAppStore((s) => s.loading)
  const streamText = useAppStore((s) => s.streamText)
  const thinkText = useAppStore((s) => s.thinkText)
  const addMessage = useAppStore((s) => s.addMessage)
  const setLoading = useAppStore((s) => s.setLoading)
  const setStreamText = useAppStore((s) => s.setStreamText)
  const appendStreamText = useAppStore((s) => s.appendStreamText)
  const setThinkText = useAppStore((s) => s.setThinkText)
  const appendThinkText = useAppStore((s) => s.appendThinkText)
  const setLastResp = useAppStore((s) => s.setLastResp)
  const sectionId = useAppStore((s) => s.sectionId)
  const modelOverride = useAppStore((s) => s.modelOverride)
  const setModelOverride = useAppStore((s) => s.setModelOverride)
  const thinkOverride = useAppStore((s) => s.thinkOverride)
  const setThinkOverride = useAppStore((s) => s.setThinkOverride)

  const [thinkOpen, setThinkOpen] = useState(false)

  const visibleModes = showAllModes ? MODES : MODES.slice(0, 4)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamText])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    addMessage({ role: 'user', content: text })
    setLoading(true)
    setStreamText('')
    setThinkText('')

    const abortCtrl = new AbortController()
    useAppStore.getState().setAbortController(abortCtrl)

    try {
      // Include editor content so the LLM can see the current document
      const docStore = useDocumentStore.getState()
      const editorText = docStore.currentDoc?.full_text || ''

      const payload: Record<string, any> = {
        message: text,
        mode,
        session_id: session?.id ?? null,
        section_id: sectionId || undefined,
      }
      // Attach current document content for modes that need it
      if (editorText) {
        payload.doc_content = editorText
        if (docStore.currentPath) payload.doc_path = docStore.currentPath
      }
      if (modelOverride) payload.model_override = modelOverride
      if (thinkOverride) payload.thinking_override = parseInt(thinkOverride)

      const res = await apiRaw('/api/chat/stream', {
        method: 'POST',
        body: JSON.stringify(payload),
        signal: abortCtrl.signal,
      })

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ error: 'Request failed' }))
        addMessage({ role: 'assistant', content: err.error ?? 'Error' })
        setLoading(false)
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let fullText = ''
      let evtType = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            evtType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && evtType) {
            try {
              const data = JSON.parse(line.slice(6))
              switch (evtType) {
                case 'thinking_start':
                  setThinkOpen(true)
                  break
                case 'thinking':
                  appendThinkText(data.text)
                  break
                case 'text':
                  fullText += data.text
                  appendStreamText(data.text)
                  break
                case 'done': {
                  const setSession = useAppStore.getState().setSession
                  if (!session && data.session_id) {
                    setSession({
                      id: data.session_id,
                      title: text.slice(0, 50),
                      mode,
                      created_at: new Date().toISOString(),
                    })
                  }

                  // If backend detected a section from the message, auto-select it
                  if (data.section_id && !useAppStore.getState().sectionId) {
                    useAppStore.getState().setSectionId(data.section_id)
                  }

                  // Auto-preview diff after draft completion
                  if (mode === 'draft' && fullText.trim()) {
                    // Use backend-detected section_id, or __full__ for whole-doc mode
                    const effectiveSectionId = useAppStore.getState().sectionId || data.section_id || '__full__'
                    useAppStore.getState().setSectionId(effectiveSectionId)
                    const previewDiff = (window as any).__previewDiff
                    if (typeof previewDiff === 'function') {
                      setTimeout(() => previewDiff(fullText), 300)
                    }
                  }
                  break
                }
                case 'error':
                  fullText += `\n\n**Error:** ${data.error}`
                  appendStreamText(`\n\n**Error:** ${data.error}`)
                  break
              }
            } catch { /* skip malformed JSON */ }
            evtType = ''
          }
        }
      }

      addMessage({ role: 'assistant', content: fullText })
      setLastResp(fullText)
      setStreamText('')
    } catch (e: any) {
      if (e.name === 'AbortError') {
        addMessage({ role: 'assistant', content: '*[Cancelled]*' })
      } else {
        addMessage({ role: 'assistant', content: `Connection error: ${e.message}` })
      }
    } finally {
      setLoading(false)
      setStreamText('')
      setThinkText('')
      useAppStore.getState().setAbortController(null)
    }
  }, [
    input, loading, mode, session, sectionId, modelOverride, thinkOverride,
    addMessage, setLoading, setStreamText, appendStreamText,
    setThinkText, appendThinkText, setLastResp,
  ])

  const handleAbort = useCallback(() => {
    const ctrl = useAppStore.getState().abortController
    if (ctrl) ctrl.abort()
  }, [])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend]
  )

  const handleAutoSync = useCallback(async () => {
    const docPath = useDocumentStore.getState().currentPath
    if (!docPath) { toast('Open a document first'); return }
    try {
      const r = await api<{ ok: boolean; sections_updated: number }>('/api/structure/auto-sync', {
        method: 'POST',
        body: JSON.stringify({ docx_path: docPath, generate_summaries: true }),
      })
      if (r.ok) {
        const s = await api('/api/structure')
        useDocumentStore.getState().setStructure(s as any)
        toast(`Synced ${r.sections_updated} sections`)
      }
    } catch (e: any) {
      toast(e.message || 'Sync failed')
    }
  }, [])

  return (
    <div className="flex h-full flex-col">
      {/* Mode pills + overrides */}
      <div className="px-2 py-1.5 border-b border-border-light space-y-1">
        <div className="flex items-center gap-1 flex-wrap">
          {visibleModes.map((m) => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`
                px-2 py-0.5 rounded-full text-[10px] font-medium transition-colors capitalize
                ${mode === m.id ? 'bg-accent text-white' : 'text-text-secondary hover:bg-bg-sidebar-hover'}
              `}
            >
              {m.label}
            </button>
          ))}
          <button
            onClick={() => setShowAllModes(!showAllModes)}
            className="text-[10px] text-text-muted hover:text-text-secondary"
          >
            {showAllModes ? 'Less' : '...'}
          </button>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={modelOverride}
            onChange={(e) => setModelOverride(e.target.value)}
            className="text-[10px] bg-transparent text-text-muted border border-border-light rounded px-1.5 py-0.5 outline-none"
          >
            {MODEL_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
          <select
            value={thinkOverride}
            onChange={(e) => setThinkOverride(e.target.value)}
            className="text-[10px] bg-transparent text-text-muted border border-border-light rounded px-1.5 py-0.5 outline-none"
          >
            {THINK_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>Think: {o.label}</option>
            ))}
          </select>
          <button
            onClick={handleAutoSync}
            title="Auto-sync structure from document"
            className="ml-auto flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary transition-colors"
          >
            <RefreshCw size={10} /> Sync
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-2">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`
              text-xs leading-relaxed rounded-lg px-3 py-2
              ${m.role === 'user'
                ? 'bg-bg-user text-text-primary ml-6'
                : 'bg-bg-card text-text-primary border border-border-light mr-2'
              }
            `}
          >
            {m.role === 'assistant' ? (
              <div
                className="prose prose-xs prose-stone max-w-none break-words
                  [&_h1]:text-sm [&_h1]:font-bold [&_h1]:mt-3 [&_h1]:mb-1
                  [&_h2]:text-xs [&_h2]:font-bold [&_h2]:mt-2 [&_h2]:mb-1
                  [&_h3]:text-xs [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-0.5
                  [&_p]:my-1 [&_p]:text-xs [&_p]:leading-relaxed
                  [&_ul]:my-1 [&_ul]:pl-4 [&_ul]:text-xs
                  [&_ol]:my-1 [&_ol]:pl-4 [&_ol]:text-xs
                  [&_li]:my-0.5
                  [&_strong]:font-semibold
                  [&_code]:bg-bg-code [&_code]:px-1 [&_code]:rounded [&_code]:text-[10px]
                  [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-2 [&_blockquote]:italic [&_blockquote]:text-text-muted
                "
                dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }}
              />
            ) : (
              <div className="whitespace-pre-wrap break-words">{m.content}</div>
            )}
            {m.role === 'assistant' && m.content.length > 50 && (
              <div className="flex items-center gap-2 mt-1.5 pt-1 border-t border-border-light">
                <button
                  onClick={() => {
                    const previewDiff = (window as any).__previewDiff
                    if (typeof previewDiff === 'function') {
                      previewDiff(m.content)
                    } else {
                      toast('Open a document first')
                    }
                  }}
                  className="text-[9px] text-accent hover:text-accent-hover transition-colors font-medium"
                  title="Preview changes in the editor"
                >
                  ↳ Apply to editor
                </button>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(m.content)
                    toast('Copied!')
                  }}
                  className="text-[9px] text-text-muted hover:text-text-secondary transition-colors"
                >
                  Copy
                </button>
              </div>
            )}
            {m.model && <div className="text-[9px] text-text-muted mt-1 text-right">{m.model}</div>}
          </div>
        ))}

        {loading && streamText && (
          <div className="text-xs leading-relaxed rounded-lg px-3 py-2 bg-bg-card border border-border-light mr-2">
            <div
              className="prose prose-xs prose-stone max-w-none break-words
                [&_h1]:text-sm [&_h1]:font-bold [&_h1]:mt-3 [&_h1]:mb-1
                [&_h2]:text-xs [&_h2]:font-bold [&_h2]:mt-2 [&_h2]:mb-1
                [&_p]:my-1 [&_p]:text-xs [&_p]:leading-relaxed
                [&_ul]:my-1 [&_ul]:pl-4 [&_ul]:text-xs
                [&_ol]:my-1 [&_ol]:pl-4 [&_ol]:text-xs
                [&_strong]:font-semibold
              "
              dangerouslySetInnerHTML={{ __html: renderMarkdown(streamText) }}
            />
          </div>
        )}

        {loading && !streamText && (
          <div className="flex items-center gap-2 px-3 py-2 text-xs text-text-muted">
            <Loader2 size={13} className="animate-spin" />
            Thinking...
          </div>
        )}

        {thinkText && (
          <div className="mx-1">
            <button
              onClick={() => setThinkOpen(!thinkOpen)}
              className="flex items-center gap-1 text-[10px] text-text-muted hover:text-text-secondary transition-colors"
            >
              <Brain size={11} />
              {thinkOpen ? 'Hide' : 'Show'} thinking
              {thinkOpen ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
            </button>
            {thinkOpen && (
              <div className="mt-1 rounded border border-border-light bg-bg-code px-2 py-1.5 text-[10px] text-text-muted leading-relaxed max-h-32 overflow-y-auto whitespace-pre-wrap">
                {thinkText}
              </div>
            )}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border-light px-2 py-2">
        <div className="flex items-end gap-1.5 rounded-lg border border-border bg-bg-card px-2 py-1.5 focus-within:border-border-focus transition-colors">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={MODES.find((m) => m.id === mode)?.hint || `Message (${mode})...`}
            rows={1}
            className="flex-1 resize-none bg-transparent text-xs text-text-primary placeholder:text-text-muted outline-none leading-relaxed max-h-24 overflow-y-auto"
          />
          {loading ? (
            <button
              onClick={handleAbort}
              className="flex items-center justify-center w-6 h-6 rounded-md bg-red text-white hover:opacity-90 transition-opacity shrink-0"
              title="Stop"
            >
              <StopCircle size={12} />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="flex items-center justify-center w-6 h-6 rounded-md bg-accent text-white hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
            >
              <Send size={12} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

/* ================================================================
   Right Panel (exported)
   ================================================================ */

export function RightPanel() {
  const [splitPct, setSplitPct] = useState(40)
  const draggingRef = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = useCallback(() => {
    draggingRef.current = true
    const handleMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current || !containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      const y = e.clientY - rect.top
      const pct = Math.min(Math.max((y / rect.height) * 100, 15), 85)
      setSplitPct(pct)
    }
    const handleMouseUp = () => {
      draggingRef.current = false
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [])

  return (
    <div ref={containerRef} className="flex h-full flex-col bg-bg-sidebar">
      <div className="overflow-hidden" style={{ height: `${splitPct}%` }}>
        <StructureTree />
      </div>
      <div
        onMouseDown={handleMouseDown}
        className="flex items-center justify-center h-1.5 cursor-row-resize hover:bg-border-light transition-colors shrink-0 group"
      >
        <div className="w-8 h-0.5 rounded-full bg-border group-hover:bg-accent-soft transition-colors" />
      </div>
      <div className="overflow-hidden border-t border-border-light" style={{ height: `${100 - splitPct}%` }}>
        <ChatPanel />
      </div>
    </div>
  )
}
