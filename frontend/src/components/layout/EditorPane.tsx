import { useCallback, useMemo } from 'react'
import type { JSONContent } from '@tiptap/react'
import {
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  FileText,
} from 'lucide-react'
import { useAppStore } from '../../stores/appStore'
import { useDocumentStore, type DocData } from '../../stores/documentStore'
import { api } from '../../lib/api'
import { docxToTiptap } from '../../lib/docxConverter'
import Editor, { type SelectionAction } from '../editor/Editor'

/**
 * Center pane that wraps the TipTap editor.
 * Until the Editor component is built, renders a placeholder area that
 * responds to document loading and shows editor content state.
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

  /* word count derived from doc metadata */
  const wordCount = currentDoc?.metadata?.word_count ?? 0

  /* current section label from structure */
  const sectionLabel = useMemo(() => {
    if (!sectionId || !structure) return null
    const sec = structure.sections.find((s) => s.id === sectionId)
    return sec?.title ?? null
  }, [sectionId, structure])

  /* progress % */
  const progressPct = structure?.progress?.pct ?? 0

  /* editor update */
  const handleEditorUpdate = useCallback(
    (content: JSONContent) => {
      setEditorContent(content)
    },
    [setEditorContent]
  )

  /* selection action — route to chat with context */
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

      // Pre-fill the chat with the selected text as context
      const prefix = action === 'find-citation'
        ? 'Find a citation to support this passage:'
        : action === 'review'
          ? 'Review and critique this passage:'
          : 'Improve and rewrite this passage:'

      addMessage({ role: 'user', content: `${prefix}\n\n"${selectedText}"` })
    },
    []
  )

  /* load document */
  const loadDocument = useCallback(
    async (path: string) => {
      setCurrentPath(path)
      try {
        const doc = await api<DocData>('/api/documents/read', {
          method: 'POST',
          body: JSON.stringify({ path }),
        })
        setCurrentDoc(doc)
        const tiptapJson = docxToTiptap(doc)
        setEditorContent(tiptapJson)
      } catch {
        setCurrentDoc(null)
        setEditorContent(null)
      }
    },
    [setCurrentPath, setCurrentDoc, setEditorContent]
  )

  return (
    <div className="flex h-full flex-col bg-bg-paper">
      {/* Top bar: file tabs + panel toggles */}
      <div className="flex items-center justify-between border-b border-border bg-bg-sidebar px-2 py-1">
        {/* Left: sidebar toggle + file info */}
        <div className="flex items-center gap-1">
          <button
            onClick={toggleSidebar}
            title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            className="flex items-center justify-center w-7 h-7 rounded text-text-secondary hover:bg-bg-sidebar-hover hover:text-text-primary transition-colors"
          >
            {sidebarOpen ? (
              <PanelLeftClose size={15} strokeWidth={1.8} />
            ) : (
              <PanelLeftOpen size={15} strokeWidth={1.8} />
            )}
          </button>

          {/* File tabs / current file */}
          {currentPath ? (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-bg-paper border border-border-light text-xs text-text-primary">
              <FileText size={12} className="text-text-muted" />
              <span className="max-w-[200px] truncate">
                {currentPath.split('/').pop()}
              </span>
            </div>
          ) : (
            /* Quick file picker when nothing is open */
            <div className="flex items-center gap-1">
              {files.slice(0, 4).map((f) => (
                <button
                  key={f.path}
                  onClick={() => loadDocument(f.path)}
                  className="flex items-center gap-1 px-2 py-1 rounded text-xs text-text-secondary hover:bg-bg-sidebar-hover transition-colors"
                >
                  <FileText size={11} className="shrink-0" />
                  <span className="max-w-[120px] truncate">{f.filename}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Right: panel toggle */}
        <button
          onClick={toggleRightPanel}
          title={rightPanelOpen ? 'Collapse panel' : 'Expand panel'}
          className="flex items-center justify-center w-7 h-7 rounded text-text-secondary hover:bg-bg-sidebar-hover hover:text-text-primary transition-colors"
        >
          {rightPanelOpen ? (
            <PanelRightClose size={15} strokeWidth={1.8} />
          ) : (
            <PanelRightOpen size={15} strokeWidth={1.8} />
          )}
        </button>
      </div>

      {/* Editor area */}
      <div className="flex-1 overflow-hidden">
        {editorContent ? (
          <Editor
            content={editorContent}
            onUpdate={handleEditorUpdate}
            editable={!useDocumentStore.getState().diffMode}
            onSelectionAction={handleSelectionAction}
          />
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center text-text-muted">
              <FileText size={40} strokeWidth={1} className="mx-auto mb-3 opacity-40" />
              <p className="text-sm">
                {files.length > 0
                  ? 'Select a document to begin editing'
                  : 'Upload a document to get started'}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Status bar */}
      <div className="flex items-center gap-4 border-t border-border bg-bg-sidebar px-4 py-1.5 text-[11px] text-text-muted">
        {sectionLabel && (
          <span className="text-text-secondary font-medium">{sectionLabel}</span>
        )}
        <span>{wordCount.toLocaleString()} words</span>
        {structure && <span>{progressPct}% complete</span>}
        {currentDoc && (
          <span className="ml-auto">
            Saved
          </span>
        )}
      </div>
    </div>
  )
}
