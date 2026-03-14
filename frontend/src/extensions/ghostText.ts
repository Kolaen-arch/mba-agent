/**
 * TipTap extension: Ghost Text (AI autocomplete).
 *
 * Displays translucent completion text ahead of the cursor.
 * Press Tab to accept, keep typing to dismiss.
 * Uses ProseMirror Decoration.widget so it doesn't corrupt document state.
 *
 * Usage:
 *   editor.commands.setGhostText('suggested completion')
 *   editor.commands.clearGhostText()
 *   editor.commands.acceptGhostText()
 */
import { Extension } from '@tiptap/core'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'

const ghostTextKey = new PluginKey('ghostText')

export const GhostText = Extension.create({
  name: 'ghostText',

  addStorage() {
    return { text: '', pos: 0 }
  },

  addCommands() {
    return {
      setGhostText:
        (text: string) =>
        ({ editor, tr }) => {
          const pos = tr.selection.from
          editor.storage.ghostText.text = text
          editor.storage.ghostText.pos = pos
          // Force re-render of decorations
          editor.view.dispatch(editor.view.state.tr.setMeta(ghostTextKey, { text, pos }))
          return true
        },

      clearGhostText:
        () =>
        ({ editor }) => {
          editor.storage.ghostText.text = ''
          editor.storage.ghostText.pos = 0
          editor.view.dispatch(editor.view.state.tr.setMeta(ghostTextKey, { text: '', pos: 0 }))
          return true
        },

      acceptGhostText:
        () =>
        ({ editor }) => {
          const { text, pos } = editor.storage.ghostText
          if (!text || !pos) return false
          editor.storage.ghostText.text = ''
          editor.storage.ghostText.pos = 0
          editor.chain().focus().insertContentAt(pos, text).run()
          return true
        },
    }
  },

  addKeyboardShortcuts() {
    return {
      Tab: ({ editor }) => {
        const { text } = editor.storage.ghostText
        if (text) {
          editor.commands.acceptGhostText()
          return true
        }
        return false
      },
    }
  },

  addProseMirrorPlugins() {
    const ext = this

    return [
      new Plugin({
        key: ghostTextKey,
        state: {
          init: () => DecorationSet.empty,
          apply(tr, oldSet) {
            const meta = tr.getMeta(ghostTextKey)
            if (meta !== undefined) {
              if (!meta.text) return DecorationSet.empty

              const widget = Decoration.widget(meta.pos, () => {
                const span = document.createElement('span')
                span.className = 'ghost-text'
                span.textContent = meta.text
                return span
              }, { side: 1 })

              return DecorationSet.create(tr.doc, [widget])
            }
            // On any other transaction (typing), clear ghost text
            if (tr.docChanged) {
              ext.storage.text = ''
              ext.storage.pos = 0
              return DecorationSet.empty
            }
            return oldSet.map(tr.mapping, tr.doc)
          },
        },
        props: {
          decorations(state) {
            return this.getState(state)
          },
        },
      }),
    ]
  },
})
