/**
 * Slash Command dropdown menu.
 *
 * Listens for 'slash-command-update' CustomEvents dispatched by the
 * SlashCommand TipTap extension and renders a floating menu at the
 * cursor position.
 */
import { useState, useEffect } from 'react'
import type { SlashCommandItem } from '../../extensions/slashCommand'

interface SlashState {
  active: boolean
  commands: SlashCommandItem[]
  selectedIndex: number
  pos: { x: number; y: number } | null
}

export default function SlashCommandMenu() {
  const [state, setState] = useState<SlashState>({
    active: false,
    commands: [],
    selectedIndex: 0,
    pos: null,
  })

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail
      setState({
        active: detail.active,
        commands: detail.commands ?? [],
        selectedIndex: detail.selectedIndex ?? 0,
        pos: detail.pos ?? null,
      })
    }
    window.addEventListener('slash-command-update', handler)
    return () => window.removeEventListener('slash-command-update', handler)
  }, [])

  if (!state.active || !state.pos || state.commands.length === 0) return null

  return (
    <div
      className="fixed z-50 bg-bg-card border border-border rounded-lg shadow-lg py-1 min-w-[200px]"
      style={{ left: state.pos.x, top: state.pos.y }}
    >
      {state.commands.map((cmd, i) => (
        <div
          key={cmd.id}
          className={`
            flex items-center gap-3 px-3 py-1.5 text-xs cursor-pointer transition-colors
            ${i === state.selectedIndex ? 'bg-bg-sidebar-hover text-text-primary' : 'text-text-secondary'}
          `}
        >
          <span className="font-mono text-accent font-medium">{cmd.label}</span>
          <span className="text-text-muted">{cmd.description}</span>
        </div>
      ))}
      <div className="px-3 py-1 text-[10px] text-text-muted border-t border-border-light mt-1 pt-1">
        ↑↓ navigate · Enter select · Esc cancel
      </div>
    </div>
  )
}
