/**
 * TipTap extension: Inline Citation Node.
 *
 * Renders citations as inline chips like [Pine & Gilmore, 1998].
 * Non-editable, styled as accent-colored pills.
 * When exported to docx, they become plain citation text.
 */
import { Node, mergeAttributes } from '@tiptap/core'

export const CitationNode = Node.create({
  name: 'citation',
  group: 'inline',
  inline: true,
  atom: true,

  addAttributes() {
    return {
      key: { default: '' },
      label: { default: '' },
    }
  },

  parseHTML() {
    return [{ tag: 'span[data-citation]' }]
  },

  renderHTML({ HTMLAttributes }) {
    return [
      'span',
      mergeAttributes(HTMLAttributes, {
        'data-citation': HTMLAttributes.key,
        class: 'citation-chip',
        contenteditable: 'false',
      }),
      `[${HTMLAttributes.label || HTMLAttributes.key}]`,
    ]
  },
})
