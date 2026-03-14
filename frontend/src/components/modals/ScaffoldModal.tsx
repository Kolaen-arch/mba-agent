/**
 * Modal for creating a new paper structure scaffold.
 * Calls POST /api/structure/scaffold
 */
import { useState, useCallback } from 'react'
import { X, Loader2 } from 'lucide-react'
import { api } from '../../lib/api'
import { useDocumentStore } from '../../stores/documentStore'
import { toast } from './Toast'

interface ScaffoldModalProps {
  open: boolean
  onClose: () => void
}

export default function ScaffoldModal({ open, onClose }: ScaffoldModalProps) {
  const [title, setTitle] = useState('')
  const [rq, setRq] = useState('')
  const [methodology, setMethodology] = useState('Design Science Research')
  const [loading, setLoading] = useState(false)

  const handleSubmit = useCallback(async () => {
    if (!title.trim() || !rq.trim()) return
    setLoading(true)
    try {
      const res = await api<{ ok: boolean; path: string }>('/api/structure/scaffold', {
        method: 'POST',
        body: JSON.stringify({
          title: title.trim(),
          research_question: rq.trim(),
          methodology: methodology.trim(),
          overwrite: false,
        }),
      })
      if ((res as any).error) {
        toast((res as any).error)
        setLoading(false)
        return
      }
      // Reload structure
      const s = await api('/api/structure')
      useDocumentStore.getState().setStructure(s as any)
      toast('Paper structure created!')
      onClose()
    } catch (e: any) {
      toast(e.message || 'Failed to create scaffold')
    } finally {
      setLoading(false)
    }
  }, [title, rq, methodology, onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="bg-bg-card rounded-xl shadow-xl border border-border w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-text-primary">Create Paper Structure</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors">
            <X size={16} />
          </button>
        </div>

        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Paper Title</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., Psychological Ownership in Experience Economy"
              className="w-full px-3 py-2 text-xs rounded-md border border-border bg-bg-paper text-text-primary placeholder:text-text-muted outline-none focus:border-border-focus transition-colors"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Research Question</label>
            <textarea
              value={rq}
              onChange={(e) => setRq(e.target.value)}
              placeholder="What is the primary research question?"
              rows={3}
              className="w-full px-3 py-2 text-xs rounded-md border border-border bg-bg-paper text-text-primary placeholder:text-text-muted outline-none focus:border-border-focus transition-colors resize-none"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Methodology</label>
            <select
              value={methodology}
              onChange={(e) => setMethodology(e.target.value)}
              className="w-full px-3 py-2 text-xs rounded-md border border-border bg-bg-paper text-text-primary outline-none focus:border-border-focus transition-colors"
            >
              <option>Design Science Research</option>
              <option>Case Study</option>
              <option>Qualitative</option>
              <option>Quantitative</option>
              <option>Mixed Methods</option>
              <option>Literature Review</option>
            </select>
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-5">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!title.trim() || !rq.trim() || loading}
            className="flex items-center gap-1.5 px-4 py-1.5 text-xs rounded-md bg-accent text-white hover:bg-accent-hover disabled:opacity-40 transition-colors"
          >
            {loading && <Loader2 size={12} className="animate-spin" />}
            Create Structure
          </button>
        </div>
      </div>
    </div>
  )
}
