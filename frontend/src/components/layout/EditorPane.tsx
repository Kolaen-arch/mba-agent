import { useCallback, useMemo, useState, useRef, useEffect } from 'react'
import type { JSONContent } from '@tiptap/react'
import {
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  FileText,
  Download,
  BarChart3,
  GitBranch,
  Zap,
} from 'lucide-react'
import { useAppStore } from '../../stores/appStore'
import { useDocumentStore, type DocData } from '../../stores/documentStore'
import { api } from '../../lib/api'
import { docxToTiptap } from '../../lib/docxConverter'
import { TEMPLATES } from '../../lib/constants'
import Editor, { type SelectionAction } from '../editor/Editor'
import DiffBar from '../editor/DiffBar'
import ScaffoldModal from '../modals/ScaffoldModal'
import BudgetModal from '../modals/BudgetModal'
import RedThreadModal from '../modals/RedThreadModal'
import { toast } from '../modals/Toast'
import type { SlashCommandItem } from '../../extensions/slashCommand'

/**
 * Center pane — wraps TipTap editor with diff workflow, templates, and modals.
 */
export function EditorPane() {
  const sidebarOpen = useAppStore((s) => s.sidebarOpen)
  const rightPanelOpen = useAppStore((s) => s.rightPanelOpen)
  const toggleSidebar = useAppStore((s) => s.toggleSidebar)
  const toggleRightPanel = useAppStore((s) => s.toggleRightPanel)
  const sectionId = useAppStore((s) => s.sectionId)

  const currentPath = useDocumentStore((s) => s.currentPath)
  const currentDoc = useDocumentStore((s) => s.currentDoc)
  const editorContent = useDocumentStore((s) => s.editorContent)
  const setCurrentPath = useDocumentStore((s) => s.setCurrentPath)
  const setCurrentDoc = useDocumentStore((s) => s.setCurrentDoc)
  const setEditorContent = useDocumentStore((s) => s.setEditorContent)
  const structure = useDocumentStore((s) => s.structure)
  const files = useDocumentStore((s) => s.files)
  const diffMode = useDocumentStore((s) => s.diffMode)
  const setDiffMode = useDocumentStore((s) => s.setDiffMode)
  const setPendingDraft = useDocumentStore((s) => s.setPendingDraft)

  // Diff state
  const [diffHtml, setDiffHtml] = useState<string | null>(null)
  const [diffStats, setDiffStats] = useState<{ currentWords: number; newWords: number; delta: number } | null>(null)

  // Document dropdown
  const [docDropdownOpen, setDocDropdownOpen] = useState(false)
  const docDropdownRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (docDropdownRef.current && !docDropdownRef.current.contains(e.target as Node)) {
        setDocDropdownOpen(false)
      }
    }
    if (docDropdownOpen) document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [docDropdownOpen])

  // Modals
  const [scaffoldOpen, setScaffoldOpen] = useState(false)
  const [budgetOpen, setBudgetOpen] = useState(false)
  const [redThreadOpen, setRedThreadOpen] = useState(false)

  const wordCount = currentDoc?.metadata?.word_count ?? 0

  const sectionLabel = useMemo(() => {
    if (!sectionId) return null
    if (sectionId === '__full__') return 'Full Document'
    if (!structure) return null
    const sec = structure.sections.find((s) => s.id === sectionId)
    return sec?.title ?? null
  }, [sectionId, structure])

  const progressPct = structure?.progress?.pct ?? 0

  const handleEditorUpdate = useCallback(
    (content: JSONContent) => {
      setEditorContent(content)
    },
    [setEditorContent]
  )

  /* selection action → route to chat with context */
  const handleSelectionAction = useCallback(
    (action: SelectionAction, selectedText: string) => {
      const setMode = useAppStore.getState().setMode
      const addMessage = useAppStore.getState().addMessage

      const modeMap: Record<SelectionAction, string> = {
        'review': 'review',
        'improve': 'draft',
        'find-citation': 'cite',
      }
      setMode(modeMap[action] as any)

      const prefix = action === 'find-citation'
        ? 'Find a citation to support this passage:'
        : action === 'review'
          ? 'Review and critique this passage:'
          : 'Improve and rewrite this passage:'

      addMessage({ role: 'user', content: `${prefix}\n\n"${selectedText}"` })

      if (!useAppStore.getState().rightPanelOpen) {
        useAppStore.getState().toggleRightPanel()
      }
    },
    []
  )

  /* slash command handler */
  const handleSlashCommand = useCallback(
    (command: SlashCommandItem, context: string) => {
      useAppStore.getState().setMode(command.mode as any)
      useAppStore.getState().addMessage({
        role: 'user',
        content: `[/${command.id}] ${context.slice(0, 300)}`,
      })
      if (!useAppStore.getState().rightPanelOpen) {
        useAppStore.getState().toggleRightPanel()
      }
    },
    []
  )

  /* load document */
  const loadDocument = useCallback(
    async (path: string) => {
      setCurrentPath(path)
      setDiffHtml(null)
      setDiffStats(null)
      setDiffMode(false)
      try {
        const doc = await api<DocData>('/api/documents/read', {
          method: 'POST',
          body: JSON.stringify({ path }),
        })
        setCurrentDoc(doc)
        setEditorContent(docxToTiptap(doc))
      } catch {
        setCurrentDoc(null)
        setEditorContent(null)
      }
    },
    [setCurrentPath, setCurrentDoc, setEditorContent, setDiffMode]
  )

  /* preview diff — called after draft completion */
  const handlePreviewDiff = useCallback(
    async (newText: string) => {
      // Read sectionId fresh from store (may have been auto-set by the done event)
      const effectiveSectionId = useAppStore.getState().sectionId
      const effectivePath = useDocumentStore.getState().currentPath
      if (!effectivePath) {
        toast('Open a document first')
        return
      }
      if (!effectiveSectionId) {
        toast('Select a section or mention one in your message (e.g. "section 1.3")')
        return
      }
      try {
        const d = await api<{ diff_html: string; current_words: number; new_words: number; delta: number }>(
          '/api/documents/preview-diff',
          {
            method: 'POST',
            body: JSON.stringify({ section_id: effectiveSectionId, new_text: newText, docx_path: effectivePath }),
          }
        )
        setDiffHtml(d.diff_html)
        setDiffStats({ currentWords: d.current_words, newWords: d.new_words, delta: d.delta })
        setDiffMode(true)
        setPendingDraft(newText)
      } catch (e: any) {
        toast(e.message || 'Diff failed')
      }
    },
    [setDiffMode, setPendingDraft]
  )

  /* accept draft */
  const handleAcceptDraft = useCallback(async () => {
    const { pendingDraft, currentPath } = useDocumentStore.getState()
    const { sectionId } = useAppStore.getState()
    if (!pendingDraft) { toast('No draft to accept'); return }
    if (!currentPath) { toast('No document open'); return }
    if (!sectionId) { toast('No section selected'); return }

    try {
      await api('/api/documents/accept-draft', {
        method: 'POST',
        body: JSON.stringify({ section_id: sectionId, new_text: pendingDraft, docx_path: currentPath }),
      })
      setPendingDraft('')
      setDiffMode(false)
      setDiffHtml(null)
      setDiffStats(null)
      await loadDocument(currentPath)
      const s = await api('/api/structure')
      useDocumentStore.getState().setStructure(s as any)
      toast('Draft accepted!')
    } catch (e: any) {
      toast(e.message || 'Accept failed')
    }
  }, [loadDocument, setDiffMode, setPendingDraft])

  /* reject draft */
  const handleRejectDraft = useCallback(() => {
    setPendingDraft('')
    setDiffMode(false)
    setDiffHtml(null)
    setDiffStats(null)
    const doc = useDocumentStore.getState().currentDoc
    if (doc) {
      setEditorContent(docxToTiptap(doc))
    }
  }, [setDiffMode, setPendingDraft, setEditorContent])

  /* template action */
  const handleTemplate = useCallback(async (templateId: string) => {
    const secId = useAppStore.getState().sectionId
    try {
      const res = await api<{ mode: string; section_id: string; prompt: string }>('/api/templates/apply', {
        method: 'POST',
        body: JSON.stringify({ template_id: templateId, section_id: secId || undefined }),
      })
      useAppStore.getState().setMode(res.mode as any)
      if (res.section_id) useAppStore.getState().setSectionId(res.section_id)
      useAppStore.getState().addMessage({ role: 'user', content: res.prompt })
      if (!useAppStore.getState().rightPanelOpen) {
        useAppStore.getState().toggleRightPanel()
      }
    } catch (e: any) {
      toast(e.message || 'Template failed')
    }
  }, [])

  // Expose previewDiff for chat panel to call after draft completion
  ;(window as any).__previewDiff = handlePreviewDiff

  return (
    <div className="flex h-full flex-col bg-bg-paper">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-border bg-bg-sidebar px-2 py-1">
        {/* Left: sidebar toggle + file info */}
        <div className="flex items-center gap-1">
          <button
            onClick={toggleSidebar}
            title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            className="flex items-center justify-center w-7 h-7 rounded text-text-secondary hover:bg-bg-sidebar-hover hover:text-text-primary transition-colors"
          >
            {sidebarOpen ? <PanelLeftClose size={15} strokeWidth={1.8} /> : <PanelLeftOpen size={15} strokeWidth={1.8} />}
          </button>

          <div className="relative">
            <button
              onClick={() => setDocDropdownOpen(!docDropdownOpen)}
              className="flex items-center gap-1.5 px-2 py-1 rounded bg-bg-paper border border-border-light text-xs text-text-primary hover:bg-bg-sidebar-hover transition-colors"
            >
              <FileText size={12} className="text-text-muted" />
              <span className="max-w-[200px] truncate">
                {currentPath ? currentPath.split('/').pop()?.split('\\').pop() : 'Open document...'}
              </span>
              <span className="text-text-muted text-[10px]">▾</span>
            </button>
            {docDropdownOpen && (
              <div
                ref={docDropdownRef}
                className="absolute left-0 top-full mt-1 z-50 bg-bg-card border border-border rounded-lg shadow-lg py-1 min-w-[200px]"
              >
                {files.map((f) => (
                  <button
                    key={f.path}
                    onClick={() => { loadDocument(f.path); setDocDropdownOpen(false) }}
                    className={`flex items-center gap-2 w-full px-3 py-1.5 text-xs text-left hover:bg-bg-sidebar-hover transition-colors ${
                      f.path === currentPath ? 'text-accent font-medium' : 'text-text-secondary'
                    }`}
                  >
                    <FileText size={11} className="shrink-0" />
                    <span className="truncate">{f.filename}</span>
                    <span className="ml-auto text-text-muted text-[10px]">{f.word_count?.toLocaleString() ?? ''} w</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Center: Template + tool actions */}
        <div className="flex items-center gap-1">
          {sectionId && TEMPLATES.filter((t) => t.requiresSection).slice(0, 4).map((t) => (
            <button
              key={t.id}
              onClick={() => handleTemplate(t.id)}
              className="px-2 py-0.5 rounded text-[10px] text-text-muted hover:text-text-secondary hover:bg-bg-sidebar-hover transition-colors"
              title={`Apply ${t.label} template`}
            >
              {t.label}
            </button>
          ))}

          <div className="w-px h-4 bg-border mx-1" />

          <button onClick={() => setScaffoldOpen(true)} title="Create paper structure"
            className="flex items-center justify-center w-7 h-7 rounded text-text-secondary hover:bg-bg-sidebar-hover hover:text-text-primary transition-colors">
            <Zap size={13} strokeWidth={1.8} />
          </button>
          <button onClick={() => setBudgetOpen(true)} title="Word budget"
            className="flex items-center justify-center w-7 h-7 rounded text-text-secondary hover:bg-bg-sidebar-hover hover:text-text-primary transition-colors">
            <BarChart3 size={13} strokeWidth={1.8} />
          </button>
          <button onClick={() => setRedThreadOpen(true)} title="Red thread audit"
            className="flex items-center justify-center w-7 h-7 rounded text-text-secondary hover:bg-bg-sidebar-hover hover:text-text-primary transition-colors">
            <GitBranch size={13} strokeWidth={1.8} />
          </button>

          {currentPath && (
            <a href={`/api/documents/download/${encodeURIComponent(currentPath)}`} title="Download docx"
              className="flex items-center justify-center w-7 h-7 rounded text-text-secondary hover:bg-bg-sidebar-hover hover:text-text-primary transition-colors">
              <Download size={13} strokeWidth={1.8} />
            </a>
          )}
        </div>

        {/* Right: panel toggle */}
        <button
          onClick={toggleRightPanel}
          title={rightPanelOpen ? 'Collapse panel' : 'Expand panel'}
          className="flex items-center justify-center w-7 h-7 rounded text-text-secondary hover:bg-bg-sidebar-hover hover:text-text-primary transition-colors"
        >
          {rightPanelOpen ? <PanelRightClose size={15} strokeWidth={1.8} /> : <PanelRightOpen size={15} strokeWidth={1.8} />}
        </button>
      </div>

      {/* Diff bar */}
      {diffMode && diffStats && (
        <DiffBar
          currentWords={diffStats.currentWords}
          newWords={diffStats.newWords}
          delta={diffStats.delta}
          onAccept={handleAcceptDraft}
          onReject={handleRejectDraft}
        />
      )}

      {/* Editor area */}
      <div className="flex-1 overflow-hidden">
        {editorContent || diffHtml ? (
          <Editor
            content={editorContent}
            onUpdate={handleEditorUpdate}
            editable={!diffMode}
            onSelectionAction={handleSelectionAction}
            onSlashCommand={handleSlashCommand}
            diffHtml={diffHtml}
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center text-text-muted">
              <FileText size={40} strokeWidth={1} className="mx-auto mb-3 opacity-40" />
              <p className="text-sm">
                {files.length > 0 ? 'Select a document to begin editing' : 'Upload a document to get started'}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Status bar */}
      <div className="flex items-center gap-4 border-t border-border bg-bg-sidebar px-4 py-1.5 text-[11px] text-text-muted">
        {sectionLabel && <span className="text-text-secondary font-medium">{sectionLabel}</span>}
        <span>{wordCount.toLocaleString()} words</span>
        {structure && <span>{progressPct}% complete</span>}
        {diffMode && <span className="text-accent font-medium">Draft Preview</span>}
        {currentDoc && !diffMode && <span className="ml-auto">Saved</span>}
        <span className="ml-auto text-[10px]">Ctrl+1-8 modes · Ctrl+D panel · Esc abort</span>
      </div>

      {/* Modals */}
      <ScaffoldModal open={scaffoldOpen} onClose={() => setScaffoldOpen(false)} />
      <BudgetModal open={budgetOpen} onClose={() => setBudgetOpen(false)} />
      <RedThreadModal open={redThreadOpen} onClose={() => setRedThreadOpen(false)} />
    </div>
  )
}
