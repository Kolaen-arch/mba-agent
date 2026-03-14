import type { JSONContent } from '@tiptap/react'

interface DocSection {
  heading: string
  paragraphs: string[]
}

interface DocResponse {
  full_text: string
  metadata: { word_count: number }
  sections: DocSection[]
}

/**
 * Convert Flask /api/documents/read response into TipTap JSON content.
 * Each heading becomes a heading node, each paragraph a paragraph node.
 */
export function docxToTiptap(doc: DocResponse): JSONContent {
  const content: JSONContent[] = []

  if (doc.sections?.length) {
    for (const section of doc.sections) {
      if (section.heading) {
        // Determine heading level from content (H1, H2, etc.)
        const level = detectHeadingLevel(section.heading)
        content.push({
          type: 'heading',
          attrs: { level },
          content: [{ type: 'text', text: section.heading }],
        })
      }

      for (const para of section.paragraphs) {
        if (!para.trim()) continue
        content.push({
          type: 'paragraph',
          content: parseParagraphContent(para),
        })
      }
    }
  } else if (doc.full_text) {
    // Fallback: split by double newlines
    const blocks = doc.full_text.split(/\n\n+/)
    for (const block of blocks) {
      const trimmed = block.trim()
      if (!trimmed) continue

      if (trimmed.startsWith('#')) {
        const match = trimmed.match(/^(#{1,6})\s+(.*)/)
        if (match) {
          content.push({
            type: 'heading',
            attrs: { level: match[1].length },
            content: [{ type: 'text', text: match[2] }],
          })
          continue
        }
      }

      content.push({
        type: 'paragraph',
        content: parseParagraphContent(trimmed),
      })
    }
  }

  return {
    type: 'doc',
    content: content.length > 0 ? content : [{ type: 'paragraph' }],
  }
}

function detectHeadingLevel(heading: string): number {
  // Simple heuristic based on numbering pattern
  if (/^\d+\.\d+\.\d+/.test(heading)) return 3
  if (/^\d+\.\d+/.test(heading)) return 2
  return 1
}

function parseParagraphContent(text: string): JSONContent[] {
  // For now, treat as plain text. Later can parse bold/italic/citations.
  return [{ type: 'text', text }]
}
