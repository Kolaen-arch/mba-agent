/**
 * TipTap extension: Slash Commands.
 *
 * Type /cite, /draft, or /review inline to trigger AI actions
 * without leaving the editor. Shows a suggestion dropdown.
 *
 * This uses TipTap's Suggestion utility (same pattern as Notion's slash menu).
 */
import { Extension } from '@tiptap/core'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'

export interface SlashCommandItem {
  id: string
  label: string
  description: string
  mode: string
}

const SLASH_COMMANDS: SlashCommandItem[] = [
  { id: 'draft', label: '/draft', description: 'Draft this section with AI', mode: 'draft' },
  { id: 'review', label: '/review', description: 'Review and critique text', mode: 'review' },
  { id: 'cite', label: '/cite', description: 'Find a citation for this claim', mode: 'cite' },
  { id: 'synthesize', label: '/synthesize', description: 'Synthesize across sources', mode: 'synthesize' },
  { id: 'transition', label: '/transition', description: 'Write a transition', mode: 'transition' },
  { id: 'consistency', label: '/consistency', description: 'Check terminology', mode: 'consistency' },
]

const slashKey = new PluginKey('slashCommand')

export interface SlashCommandCallbacks {
  onCommand?: (command: SlashCommandItem, context: string) => void
}

let _callbacks: SlashCommandCallbacks = {}

export function setSlashCommandCallbacks(cb: SlashCommandCallbacks) {
  _callbacks = cb
}

export const SlashCommand = Extension.create({
  name: 'slashCommand',

  addStorage() {
    return {
      active: false,
      query: '',
      menuPos: null as { x: number; y: number } | null,
      filteredCommands: [] as SlashCommandItem[],
      selectedIndex: 0,
      startPos: 0,
    }
  },

  addProseMirrorPlugins() {
    const ext = this

    return [
      new Plugin({
        key: slashKey,

        props: {
          handleTextInput(view, from, to, text) {
            const state = view.state
            const $from = state.doc.resolve(from)
            const lineText = $from.parent.textBetween(
              Math.max(0, $from.parentOffset - 20),
              $from.parentOffset,
              ''
            ) + text

            // Check if user is typing a slash command
            const slashMatch = lineText.match(/\/(\w*)$/)
            if (slashMatch) {
              const query = slashMatch[1].toLowerCase()
              const filtered = SLASH_COMMANDS.filter(
                (c) => c.id.startsWith(query) || c.label.includes('/' + query)
              )

              if (filtered.length > 0) {
                ext.storage.active = true
                ext.storage.query = query
                ext.storage.filteredCommands = filtered
                ext.storage.selectedIndex = 0
                ext.storage.startPos = from - slashMatch[0].length + 1 // +1 because text hasn't been inserted yet

                // Get position for popup
                const coords = view.coordsAtPos(from)
                ext.storage.menuPos = { x: coords.left, y: coords.bottom + 4 }

                // Dispatch custom event for React to listen to
                window.dispatchEvent(
                  new CustomEvent('slash-command-update', {
                    detail: {
                      active: true,
                      commands: filtered,
                      selectedIndex: 0,
                      pos: ext.storage.menuPos,
                    },
                  })
                )
              } else {
                ext.storage.active = false
                window.dispatchEvent(
                  new CustomEvent('slash-command-update', { detail: { active: false } })
                )
              }
            } else if (ext.storage.active) {
              ext.storage.active = false
              window.dispatchEvent(
                new CustomEvent('slash-command-update', { detail: { active: false } })
              )
            }

            return false
          },

          handleKeyDown(view, event) {
            if (!ext.storage.active) return false

            if (event.key === 'ArrowDown') {
              event.preventDefault()
              ext.storage.selectedIndex = Math.min(
                ext.storage.selectedIndex + 1,
                ext.storage.filteredCommands.length - 1
              )
              window.dispatchEvent(
                new CustomEvent('slash-command-update', {
                  detail: {
                    active: true,
                    commands: ext.storage.filteredCommands,
                    selectedIndex: ext.storage.selectedIndex,
                    pos: ext.storage.menuPos,
                  },
                })
              )
              return true
            }

            if (event.key === 'ArrowUp') {
              event.preventDefault()
              ext.storage.selectedIndex = Math.max(ext.storage.selectedIndex - 1, 0)
              window.dispatchEvent(
                new CustomEvent('slash-command-update', {
                  detail: {
                    active: true,
                    commands: ext.storage.filteredCommands,
                    selectedIndex: ext.storage.selectedIndex,
                    pos: ext.storage.menuPos,
                  },
                })
              )
              return true
            }

            if (event.key === 'Enter') {
              event.preventDefault()
              const cmd = ext.storage.filteredCommands[ext.storage.selectedIndex]
              if (cmd) {
                // Delete the slash command text
                const { state } = view
                const pos = state.selection.from
                // Find the start of the slash command
                const $pos = state.doc.resolve(pos)
                const lineText = $pos.parent.textBetween(0, $pos.parentOffset, '')
                const slashIdx = lineText.lastIndexOf('/')
                if (slashIdx >= 0) {
                  const deleteFrom = $pos.start() + slashIdx
                  const tr = state.tr.delete(deleteFrom, pos)
                  view.dispatch(tr)
                }

                // Get surrounding context for the command
                const context = state.doc.textBetween(
                  Math.max(0, pos - 500),
                  Math.min(state.doc.content.size, pos + 200),
                  '\n'
                )

                ext.storage.active = false
                window.dispatchEvent(
                  new CustomEvent('slash-command-update', { detail: { active: false } })
                )

                _callbacks.onCommand?.(cmd, context)
              }
              return true
            }

            if (event.key === 'Escape') {
              ext.storage.active = false
              window.dispatchEvent(
                new CustomEvent('slash-command-update', { detail: { active: false } })
              )
              return true
            }

            return false
          },
        },
      }),
    ]
  },
})
