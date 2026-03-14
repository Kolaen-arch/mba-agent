/**
 * Diff action bar — appears above the editor when in diff mode.
 * Shows word count delta and accept/reject buttons.
 */
import { Check, X, ArrowUpDown } from 'lucide-react'

interface DiffBarProps {
  currentWords: number
  newWords: number
  delta: number
  onAccept: () => void
  onReject: () => void
}

export default function DiffBar({ currentWords, newWords, delta, onAccept, onReject }: DiffBarProps) {
  return (
    <div className="flex items-center justify-between px-4 py-2 bg-bg-code border-b border-border">
      <div className="flex items-center gap-3 text-xs">
        <ArrowUpDown size={14} className="text-accent" />
        <span className="text-text-secondary font-medium">Draft Changes</span>
        <span className="text-text-muted">
          {currentWords} → {newWords} words
        </span>
        <span className={delta >= 0 ? 'text-green font-medium' : 'text-red font-medium'}>
          ({delta >= 0 ? '+' : ''}{delta})
        </span>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onReject}
          className="flex items-center gap-1.5 px-3 py-1 rounded-md text-xs text-text-secondary border border-border hover:bg-bg-sidebar-hover transition-colors"
        >
          <X size={13} />
          Reject
        </button>
        <button
          onClick={onAccept}
          className="flex items-center gap-1.5 px-3 py-1 rounded-md text-xs text-white bg-green hover:opacity-90 transition-opacity"
        >
          <Check size={13} />
          Accept
        </button>
      </div>
    </div>
  )
}
