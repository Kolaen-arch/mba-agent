/**
 * TipTap extension: Track Changes decorations.
 *
 * When diff mode is active, the editor content is set to the diff HTML
 * from Flask's compute_inline_diff(). This extension adds decoration
 * marks for <ins> and <del> elements so they render with the correct
 * styles (green background / red strikethrough).
 *
 * This is NOT a full OT-based track changes system — it simply renders
 * the server-side diff for accept/reject workflow.
 */
import { Mark, mergeAttributes } from '@tiptap/core'

export const InsertionMark = Mark.create({
  name: 'insertion',
  parseHTML() {
    return [{ tag: 'ins' }]
  },
  renderHTML({ HTMLAttributes }) {
    return ['ins', mergeAttributes(HTMLAttributes), 0]
  },
})

export const DeletionMark = Mark.create({
  name: 'deletion',
  parseHTML() {
    return [{ tag: 'del' }]
  },
  renderHTML({ HTMLAttributes }) {
    return ['del', mergeAttributes(HTMLAttributes), 0]
  },
})
