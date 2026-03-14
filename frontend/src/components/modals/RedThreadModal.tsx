/**
 * Red Thread Audit modal.
 * Calls GET /api/structure/red-thread-audit and displays coherence results.
 */
import { useState, useEffect } from 'react'
import { X, Loader2, AlertTriangle, CheckCircle2 } from 'lucide-react'
import { api } from '../../lib/api'

interface AuditResult {
  from_section: string
  to_section: string
  coherence: number
  red_thread: string
  suggestion: string
}

interface AuditData {
  red_thread: string
  total_pairs: number
  average_coherence: number
  weak_transitions: number
  results: AuditResult[]
}

interface RedThreadModalProps {
  open: boolean
  onClose: () => void
}

export default function RedThreadModal({ open, onClose }: RedThreadModalProps) {
  const [data, setData] = useState<AuditData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!open) return
    setLoading(true)
    setError('')
    api<AuditData>('/api/structure/red-thread-audit')
      .then(setData)
      .catch((e) => setError(e.message || 'Failed to run audit'))
      .finally(() => setLoading(false))
  }, [open])

  if (!open) return null

  const getThreadColor = (thread: string) => {
    if (thread === 'visible') return 'text-green'
    if (thread === 'partial') return 'text-accent'
    return 'text-red'
  }

  const getCoherenceColor = (score: number) => {
    if (score >= 4) return 'text-green'
    if (score >= 3) return 'text-accent'
    return 'text-red'
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="bg-bg-card rounded-xl shadow-xl border border-border w-full max-w-2xl max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between px-6 py-4 border-b border-border shrink-0">
          <h2 className="text-sm font-semibold text-text-primary">Red Thread Audit</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary transition-colors">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="flex items-center justify-center py-8 gap-2 text-xs text-text-muted">
              <Loader2 size={16} className="animate-spin" />
              Running coherence audit (this may take a moment)...
            </div>
          )}

          {error && (
            <div className="text-xs text-red text-center py-8">{error}</div>
          )}

          {data && (
            <>
              {/* Summary */}
              <div className="rounded-lg bg-bg-paper border border-border-light p-4 mb-5">
                <div className="text-xs text-text-muted mb-2">Red Thread</div>
                <div className="text-sm text-text-primary font-medium italic mb-3">"{data.red_thread}"</div>
                <div className="flex items-center gap-4 text-xs">
                  <span className="text-text-secondary">
                    Avg coherence: <span className={getCoherenceColor(data.average_coherence)}>{data.average_coherence.toFixed(1)}/5</span>
                  </span>
                  <span className="text-text-secondary">
                    Weak transitions: <span className={data.weak_transitions > 0 ? 'text-red font-medium' : 'text-green'}>{data.weak_transitions}</span>
                  </span>
                  <span className="text-text-muted">{data.total_pairs} pairs analyzed</span>
                </div>
              </div>

              {/* Results */}
              <div className="space-y-2">
                {data.results.map((r, i) => (
                  <div
                    key={i}
                    className="rounded-md border border-border-light p-3 hover:bg-bg-paper transition-colors"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-text-primary font-medium">
                        {r.from_section} → {r.to_section}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className={`text-[10px] font-medium uppercase ${getThreadColor(r.red_thread)}`}>
                          {r.red_thread}
                        </span>
                        <span className={`text-xs font-medium ${getCoherenceColor(r.coherence)}`}>
                          {r.coherence}/5
                        </span>
                        {r.coherence >= 4 ? (
                          <CheckCircle2 size={12} className="text-green" />
                        ) : (
                          <AlertTriangle size={12} className={r.coherence >= 3 ? 'text-accent' : 'text-red'} />
                        )}
                      </div>
                    </div>
                    {r.suggestion && (
                      <div className="text-[11px] text-text-muted mt-1">{r.suggestion}</div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
