import { useCallback } from 'react'
import { api } from '../lib/api'
import { useDocumentStore } from '../stores/documentStore'
import { useAppStore } from '../stores/appStore'
import { docxToTiptap } from '../lib/docxConverter'

export function useDocument() {
  const {
    setFiles, setCurrentPath, setCurrentDoc,
    setEditorContent, setPendingDraft, setDiffMode, setStructure,
  } = useDocumentStore()

  const loadFiles = useCallback(async () => {
    try {
      const files = await api('/api/documents')
      setFiles(files)
    } catch (e) {
      console.error('Failed to load files:', e)
    }
  }, [])

  const openDocument = useCallback(async (path: string) => {
    try {
      const doc = await api('/api/documents/read', {
        method: 'POST',
        body: JSON.stringify({ path }),
      })
      if (doc.error) throw new Error(doc.error)

      setCurrentPath(path)
      setCurrentDoc(doc)
      setEditorContent(docxToTiptap(doc))
      setDiffMode(false)
      setPendingDraft('')

      // Auto-sync if structure exists
      const structure = useDocumentStore.getState().structure
      if (structure?.sections?.length) {
        try {
          const r = await api('/api/structure/auto-sync', {
            method: 'POST',
            body: JSON.stringify({ docx_path: path, generate_summaries: false }),
          })
          if (r.ok && r.sections_updated > 0) {
            await loadStructure()
          }
        } catch {}
      }

      return doc
    } catch (e) {
      console.error('Failed to open document:', e)
      throw e
    }
  }, [])

  const previewDiff = useCallback(async (newText: string) => {
    const { currentPath } = useDocumentStore.getState()
    const { sectionId } = useAppStore.getState()
    if (!currentPath || !sectionId) return null

    try {
      const d = await api('/api/documents/preview-diff', {
        method: 'POST',
        body: JSON.stringify({
          section_id: sectionId,
          new_text: newText,
          docx_path: currentPath,
        }),
      })
      if (d.error) throw new Error(d.error)
      setPendingDraft(newText)
      setDiffMode(true)
      return d
    } catch (e) {
      console.error('Diff failed:', e)
      return null
    }
  }, [])

  const acceptDraft = useCallback(async () => {
    const { pendingDraft, currentPath } = useDocumentStore.getState()
    const { sectionId } = useAppStore.getState()
    if (!pendingDraft || !currentPath || !sectionId) return

    try {
      const d = await api('/api/documents/accept-draft', {
        method: 'POST',
        body: JSON.stringify({
          section_id: sectionId,
          new_text: pendingDraft,
          docx_path: currentPath,
        }),
      })
      if (d.error) throw new Error(d.error)
      setPendingDraft('')
      setDiffMode(false)
      // Reload document
      await openDocument(currentPath)
      await loadStructure()
      return d
    } catch (e) {
      console.error('Accept failed:', e)
    }
  }, [])

  const rejectDraft = useCallback(() => {
    setPendingDraft('')
    setDiffMode(false)
    // Restore original content
    const doc = useDocumentStore.getState().currentDoc
    if (doc) {
      setEditorContent(docxToTiptap(doc))
    }
  }, [])

  const loadStructure = useCallback(async () => {
    try {
      const s = await api('/api/structure')
      setStructure(s)
      return s
    } catch {
      setStructure(null)
      return null
    }
  }, [])

  const syncSection = useCallback(async (sectionIdOverride?: string) => {
    const { currentPath } = useDocumentStore.getState()
    const secId = sectionIdOverride || useAppStore.getState().sectionId
    if (!currentPath || !secId) return

    try {
      const r = await api('/api/structure/sync-docx', {
        method: 'POST',
        body: JSON.stringify({ section_id: secId, docx_path: currentPath }),
      })
      if (r.ok) await loadStructure()
      return r
    } catch (e) {
      console.error('Sync failed:', e)
    }
  }, [])

  return {
    loadFiles,
    openDocument,
    previewDiff,
    acceptDraft,
    rejectDraft,
    loadStructure,
    syncSection,
  }
}
