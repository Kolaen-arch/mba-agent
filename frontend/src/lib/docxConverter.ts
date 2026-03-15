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

/**
 * Convert TipTap JSON content back to markdown text for saving to docx.
 * Headings become # / ## / ###, paragraphs become plain text.
 */
export function tiptapToMarkdown(doc: JSONContent): string {
  if (!doc.content) return ''

  const lines: string[] = []
  for (const node of doc.content) {
    if (node.type === 'heading') {
      const level = node.attrs?.level ?? 1
      const prefix = '#'.repeat(level)
      const text = extractText(node)
      lines.push(`${prefix} ${text}`)
      lines.push('')
    } else if (node.type === 'paragraph') {
      const text = extractText(node)
      lines.push(text)
      lines.push('')
    } else {
      // Other node types: extract text as-is
      const text = extractText(node)
      if (text) {
        lines.push(text)
        lines.push('')
      }
    }
  }
  return lines.join('\n').trim()
}

function extractText(node: JSONContent): string {
  if (node.text) return node.text
  if (!node.content) return ''
  return node.content.map(extractText).join('')
}
