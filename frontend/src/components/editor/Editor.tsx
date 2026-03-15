import { useEditor, EditorContent, type JSONContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import { Underline } from '@tiptap/extension-underline'
import { TextAlign } from '@tiptap/extension-text-align'
import { Highlight } from '@tiptap/extension-highlight'
import { Placeholder } from '@tiptap/extension-placeholder'
import { Table } from '@tiptap/extension-table'
import { TableRow } from '@tiptap/extension-table-row'
import { TableCell } from '@tiptap/extension-table-cell'
import { TableHeader } from '@tiptap/extension-table-header'
import { Typography } from '@tiptap/extension-typography'
import { useEffect, useCallback, useRef, useState } from 'react'
import { MessageSquareMore, Sparkles, BookOpen } from 'lucide-react'
import EditorToolbar from './EditorToolbar'
import SlashCommandMenu from './SlashCommandMenu'
import { InsertionMark, DeletionMark } from '../../extensions/trackChanges'
import { CitationNode } from '../../extensions/citationNode'
import { SectionBreak } from '../../extensions/sectionBreak'
import { GhostText } from '../../extensions/ghostText'
import { SlashCommand, setSlashCommandCallbacks, type SlashCommandItem } from '../../extensions/slashCommand'

export type SelectionAction = 'review' | 'improve' | 'find-citation'

interface EditorProps {
  content: JSONContent | null
  onUpdate: (content: JSONContent) => void
  editable?: boolean
  onSelectionAction?: (action: SelectionAction, selectedText: string) => void
  onSlashCommand?: (command: SlashCommandItem, context: string) => void
  diffHtml?: string | null
}

export default function Editor({
  content,
  onUpdate,
  editable = true,
  onSelectionAction,
  onSlashCommand,
  diffHtml,
}: EditorProps) {
  const [menuPos, setMenuPos] = useState<{ x: number; y: number } | null>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  // Wire up slash command callbacks
  useEffect(() => {
    if (onSlashCommand) {
      setSlashCommandCallbacks({ onCommand: onSlashCommand })
    }
    return () => setSlashCommandCallbacks({})
  }, [onSlashCommand])

  const editor = useEditor({
    extensions: [
      StarterKit.configure({ heading: { levels: [1, 2, 3] } }),
      Underline,
      TextAlign.configure({ types: ['heading', 'paragraph'] }),
      Highlight.configure({ multicolor: false }),
      Placeholder.configure({ placeholder: 'Start writing your paper...' }),
      Table.configure({ resizable: true }),
      TableRow,
      TableCell,
      TableHeader,
      Typography,
      // Track changes marks (Phase 3)
      InsertionMark,
      DeletionMark,
      // AI extensions (Phase 6)
      CitationNode,
      SectionBreak,
      GhostText,
      SlashCommand,
    ],
    editable,
    content: content ?? undefined,
    onUpdate: ({ editor: ed }) => {
      onUpdate(ed.getJSON())
    },
    editorProps: {
      attributes: { class: 'tiptap' },
    },
    onSelectionUpdate: ({ editor: ed }) => {
      const { from, to } = ed.state.selection
      const text = ed.state.doc.textBetween(from, to, ' ')
      if (text.trim().length > 2 && ed.isEditable) {
        const coords = ed.view.coordsAtPos(from)
        setMenuPos({ x: coords.left, y: coords.top - 45 })
      } else {
        setMenuPos(null)
      }
    },
  })

  // Sync editable
  useEffect(() => {
    if (editor && editor.isEditable !== editable) {
      editor.setEditable(editable)
    }
  }, [editor, editable])

  // Sync content from props (normal mode)
  useEffect(() => {
    if (!editor || !content || diffHtml) return
    const currentJSON = JSON.stringify(editor.getJSON())
    const incomingJSON = JSON.stringify(content)
    if (currentJSON !== incomingJSON) {
      editor.commands.setContent(content)
    }
  }, [editor, content, diffHtml])

  // In diff mode, hide the editor and show the diff overlay
  useEffect(() => {
    if (!editor) return
    if (diffHtml) {
      editor.setEditable(false)
    } else {
      editor.setEditable(editable)
    }
  }, [editor, diffHtml, editable])

  // Close menu on click outside
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuPos(null)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const handleSelectionAction = useCallback(
    (action: SelectionAction) => {
      if (!editor || !onSelectionAction) return
      const { from, to } = editor.state.selection
      const selectedText = editor.state.doc.textBetween(from, to, ' ')
      if (selectedText.trim()) {
        onSelectionAction(action, selectedText)
        setMenuPos(null)
      }
    },
    [editor, onSelectionAction],
  )

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <EditorToolbar editor={editor} />

      {/* Scrollable document area with "page" appearance */}
      <div className="flex-1 overflow-y-auto bg-bg py-8 px-4 relative">
        <div
          className="
            relative mx-auto
            max-w-[800px] min-h-[1056px]
            bg-bg-card
            rounded shadow-[0_1px_4px_rgba(0,0,0,0.08)]
            border border-border-light
          "
        >
          {diffHtml ? (
            <div
              className="
                p-8 font-serif text-sm leading-relaxed text-text-primary
                [&_del]:bg-red/15 [&_del]:text-red [&_del]:line-through [&_del]:px-0.5 [&_del]:rounded
                [&_ins]:bg-green/15 [&_ins]:text-green [&_ins]:no-underline [&_ins]:px-0.5 [&_ins]:rounded
                [&_p]:my-2 [&_h1]:text-lg [&_h1]:font-bold [&_h1]:mt-4 [&_h1]:mb-2
                [&_h2]:text-base [&_h2]:font-bold [&_h2]:mt-3 [&_h2]:mb-2
                [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-1
              "
              dangerouslySetInnerHTML={{ __html: diffHtml }}
            />
          ) : (
            <EditorContent editor={editor} />
          )}
        </div>

        {/* Floating selection menu */}
        {menuPos && !diffHtml && (
          <div
            ref={menuRef}
            className="fixed z-50 flex items-center gap-1 bg-bg-card border border-border rounded-lg shadow-lg px-1 py-1"
            style={{ left: menuPos.x, top: menuPos.y }}
          >
            <BubbleButton
              onClick={() => handleSelectionAction('review')}
              label="Review"
              icon={<MessageSquareMore size={13} strokeWidth={1.8} />}
            />
            <BubbleButton
              onClick={() => handleSelectionAction('improve')}
              label="Improve"
              icon={<Sparkles size={13} strokeWidth={1.8} />}
            />
            <BubbleButton
              onClick={() => handleSelectionAction('find-citation')}
              label="Find Citation"
              icon={<BookOpen size={13} strokeWidth={1.8} />}
            />
          </div>
        )}

        {/* Slash command dropdown */}
        <SlashCommandMenu />
      </div>
    </div>
  )
}

/* ---------- Internal sub-component ---------- */

interface BubbleButtonProps {
  onClick: () => void
  label: string
  icon: React.ReactNode
}

function BubbleButton({ onClick, label, icon }: BubbleButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="
        flex items-center gap-1.5
        px-2.5 py-1.5
        rounded-md text-xs
        text-text-secondary
        hover:bg-bg-sidebar-hover hover:text-accent
        transition-colors duration-100
      "
    >
      {icon}
      <span>{label}</span>
    </button>
  )
}
