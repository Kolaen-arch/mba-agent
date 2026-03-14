/**
 * TipTap extension: Section Break node.
 *
 * Renders a visual section boundary marker in the editor,
 * showing the section title and word count target.
 * Non-editable, acts as a separator between document sections.
 */
import { Node, mergeAttributes } from '@tiptap/core'

export const SectionBreak = Node.create({
  name: 'sectionBreak',
  group: 'block',
  atom: true,
  selectable: true,
  draggable: false,

  addAttributes() {
    return {
      sectionId: { default: '' },
      title: { default: '' },
      targetWords: { default: 0 },
      wordCount: { default: 0 },
    }
  },

  parseHTML() {
    return [{ tag: 'div[data-section-break]' }]
  },

  renderHTML({ HTMLAttributes }) {
    const pct = HTMLAttributes.targetWords
      ? Math.min(100, Math.round((HTMLAttributes.wordCount / HTMLAttributes.targetWords) * 100))
      : 0

    return [
      'div',
      mergeAttributes(HTMLAttributes, {
        'data-section-break': HTMLAttributes.sectionId,
        class: 'section-break',
        contenteditable: 'false',
      }),
      [
        'div',
        { class: 'section-break-content' },
        [
          'span',
          { class: 'section-break-title' },
          `§ ${HTMLAttributes.title}`,
        ],
        [
          'span',
          { class: 'section-break-stats' },
          `${HTMLAttributes.wordCount}/${HTMLAttributes.targetWords} words (${pct}%)`,
        ],
      ],
    ]
  },
})
