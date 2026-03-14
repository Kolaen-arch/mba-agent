/**
 * Word Budget Dashboard modal.
 * Calls GET /api/structure/word-budget and displays per-section budgets.
 */
import { useState, useEffect } from 'react'
import { X, Loader2 } from 'lucide-react'
import { api } from '../../lib/api'

interface BudgetSection {
  id: string
  title: string
  word_count: number
  target_words: number
  pct: number
  status: string
}

interface BudgetData {
  sections: BudgetSection[]
  total_words: number
  target_words: number
  total_pct: number
  page_estimate: number
  by_status: Record<string, number>
}

interface BudgetModalProps {
  open: boolean
  onClose: () => void
}

export default function BudgetModal({ open, onClose }: BudgetModalProps) {
  const [data, setData] = useState<BudgetData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!open) return
    setLoading(true)
    api<BudgetData>('/api/structure/word-budget')
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false))
  }, [open])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="bg-bg-card rounded-xl shadow-xl border border-border w-full max-w-lg max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border shrink-0">
          <h2 className="text-sm font-semibold text-text-primary">Word Budget</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 size={20} className="animate-spin text-text-muted" />
            </div>
          )}

          {data && (
            <>
              {/* Summary */}
              <div className="grid grid-cols-3 gap-3 mb-5">
                <div className="rounded-lg bg-bg-paper border border-border-light p-3 text-center">
                  <div className="text-lg font-semibold text-text-primary">
                    {data.total_words.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-text-muted">
                    / {data.target_words.toLocaleString()} target
                  </div>
                </div>
                <div className="rounded-lg bg-bg-paper border border-border-light p-3 text-center">
                  <div className="text-lg font-semibold text-text-primary">{data.total_pct}%</div>
                  <div className="text-[10px] text-text-muted">Complete</div>
                </div>
                <div className="rounded-lg bg-bg-paper border border-border-light p-3 text-center">
                  <div className="text-lg font-semibold text-text-primary">~{data.page_estimate}</div>
                  <div className="text-[10px] text-text-muted">Pages</div>
                </div>
              </div>

              {/* Status breakdown */}
              {data.by_status && (
                <div className="flex items-center gap-3 mb-4 text-[10px]">
                  {Object.entries(data.by_status).map(([status, count]) => (
                    <span key={status} className="flex items-center gap-1 text-text-muted">
                      <span className={`w-2 h-2 rounded-full ${
                        status === 'complete' || status === 'done' ? 'bg-green'
                          : status === 'drafting' || status === 'draft' ? 'bg-accent'
                          : 'bg-border'
                      }`} />
                      {status}: {count}
                    </span>
                  ))}
                </div>
              )}

              {/* Per-section table */}
              <div className="space-y-1">
                {data.sections?.map((sec) => (
                  <div
                    key={sec.id}
                    className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-bg-paper transition-colors text-xs"
                  >
                    <span className="flex-1 truncate text-text-primary">{sec.title}</span>
                    <span className="text-text-muted tabular-nums shrink-0 w-16 text-right">
                      {sec.word_count}/{sec.target_words}
                    </span>
                    <div className="w-20 h-1.5 rounded-full bg-border-light overflow-hidden shrink-0">
                      <div
                        className={`h-full rounded-full transition-all ${
                          sec.pct >= 80 ? 'bg-green' : sec.pct >= 40 ? 'bg-accent' : 'bg-border'
                        }`}
                        style={{ width: `${Math.min(sec.pct, 100)}%` }}
                      />
                    </div>
                    <span className="text-text-muted tabular-nums w-10 text-right shrink-0">
                      {sec.pct}%
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}

          {!loading && !data && (
            <div className="text-xs text-text-muted text-center py-8">
              No paper structure found. Create one first.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
