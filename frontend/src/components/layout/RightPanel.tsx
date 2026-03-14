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
} from 'lucide-react'
import { useAppStore, type Message } from '../../stores/appStore'
import { useDocumentStore, type PaperSection } from '../../stores/documentStore'
import { apiRaw } from '../../lib/api'
import { VISIBLE_MODES, type ModeId } from '../../lib/constants'

/* ================================================================
   Structure tree (top half)
   ================================================================ */

function StatusIcon({ status }: { status: string }) {
  if (status === 'complete' || status === 'done') {
    return <CheckCircle2 size={14} className="text-green shrink-0" />
  }
  if (status === 'in_progress' || status === 'in-progress' || status === 'draft') {
    return <CircleDot size={14} className="text-accent shrink-0" />
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
            onClick={(e) => {
              e.stopPropagation()
              setOpen(!open)
            }}
            className="shrink-0"
            aria-label={open ? 'Collapse' : 'Expand'}
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

      {open &&
        hasChildren &&
        childSections.map((child) => (
          <SectionNode
            key={child.id}
            section={child}
            childSections={
              useDocumentStore
                .getState()
                .structure?.sections.filter((s) => s.parent_id === child.id)
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
      <div className="flex items-center justify-center h-full text-xs text-text-muted p-4">
        No paper structure loaded.
        <br />
        Run <code className="text-text-secondary">make scaffold</code> to generate one.
      </div>
    )
  }

  const roots = structure.sections
    .filter((s) => !s.parent_id)
    .sort((a, b) => a.order - b.order)

  return (
    <div className="p-2 overflow-y-auto">
      <div className="px-2 pb-1.5 text-[10px] font-medium text-text-muted uppercase tracking-widest">
        Structure
      </div>
      {roots.map((root) => (
        <SectionNode
          key={root.id}
          section={root}
          childSections={structure.sections
            .filter((s) => s.parent_id === root.id)
            .sort((a, b) => a.order - b.order)}
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

  const [thinkOpen, setThinkOpen] = useState(false)

  /* auto-scroll */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamText])

  /* send message via SSE streaming — matches Flask /api/chat/stream protocol */
  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    addMessage({ role: 'user', content: text })
    setLoading(true)
    setStreamText('')
    setThinkText('')

    const abortCtrl = new AbortController()
    const setAbortController = useAppStore.getState().setAbortController
    setAbortController(abortCtrl)

    try {
      const res = await apiRaw('/api/chat/stream', {
        method: 'POST',
        body: JSON.stringify({
          message: text,
          mode,
          session_id: session?.id ?? null,
          section_id: sectionId || undefined,
        }),
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
                case 'thinking_done':
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
                  break
                }
                case 'error':
                  fullText += `\n\n**Error:** ${data.error}`
                  appendStreamText(`\n\n**Error:** ${data.error}`)
                  break
              }
            } catch {
              /* skip malformed JSON */
            }
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
    input, loading, mode, session, sectionId,
    addMessage, setLoading, setStreamText, appendStreamText,
    setThinkText, appendThinkText, setLastResp,
  ])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend]
  )

  return (
    <div className="flex h-full flex-col">
      {/* Mode pills */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-border-light">
        {VISIBLE_MODES.map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`
              px-2.5 py-0.5 rounded-full text-[11px] font-medium transition-colors capitalize
              ${
                mode === m
                  ? 'bg-accent text-white'
                  : 'text-text-secondary hover:bg-bg-sidebar-hover'
              }
            `}
          >
            {m}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-2">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`
              text-xs leading-relaxed rounded-lg px-3 py-2
              ${
                m.role === 'user'
                  ? 'bg-bg-user text-text-primary ml-6'
                  : 'bg-bg-card text-text-primary border border-border-light mr-2'
              }
            `}
          >
            <div className="whitespace-pre-wrap break-words">{m.content}</div>
          </div>
        ))}

        {/* Streaming indicator */}
        {loading && streamText && (
          <div className="text-xs leading-relaxed rounded-lg px-3 py-2 bg-bg-card border border-border-light mr-2">
            <div className="whitespace-pre-wrap break-words">{streamText}</div>
          </div>
        )}

        {loading && !streamText && (
          <div className="flex items-center gap-2 px-3 py-2 text-xs text-text-muted">
            <Loader2 size={13} className="animate-spin" />
            Thinking...
          </div>
        )}

        {/* Thinking box */}
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
            placeholder={`Message (${mode})...`}
            rows={1}
            className="flex-1 resize-none bg-transparent text-xs text-text-primary placeholder:text-text-muted outline-none leading-relaxed max-h-24 overflow-y-auto"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="flex items-center justify-center w-6 h-6 rounded-md bg-accent text-white hover:bg-accent-hover disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
          >
            <Send size={12} />
          </button>
        </div>
      </div>
    </div>
  )
}

/* ================================================================
   Right Panel (exported)
   ================================================================ */

export function RightPanel() {
  const [splitPct, setSplitPct] = useState(45) // top half %
  const draggingRef = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)

  /* Resizable divider */
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
      {/* Structure tree (top) */}
      <div
        className="overflow-hidden"
        style={{ height: `${splitPct}%` }}
      >
        <StructureTree />
      </div>

      {/* Resizable divider */}
      <div
        onMouseDown={handleMouseDown}
        className="flex items-center justify-center h-1.5 cursor-row-resize hover:bg-border-light transition-colors shrink-0 group"
      >
        <div className="w-8 h-0.5 rounded-full bg-border group-hover:bg-accent-soft transition-colors" />
      </div>

      {/* Chat (bottom) */}
      <div
        className="overflow-hidden border-t border-border-light"
        style={{ height: `${100 - splitPct}%` }}
      >
        <ChatPanel />
      </div>
    </div>
  )
}
