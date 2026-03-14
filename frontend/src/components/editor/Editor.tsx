import { useEditor, EditorContent, type JSONContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import { Underline } from '@tiptap/extension-underline';
import { TextAlign } from '@tiptap/extension-text-align';
import { Highlight } from '@tiptap/extension-highlight';
import { Placeholder } from '@tiptap/extension-placeholder';
import { Table } from '@tiptap/extension-table';
import { TableRow } from '@tiptap/extension-table-row';
import { TableCell } from '@tiptap/extension-table-cell';
import { TableHeader } from '@tiptap/extension-table-header';
import { Typography } from '@tiptap/extension-typography';
import { useEffect, useCallback, useRef, useState } from 'react';
import { MessageSquareMore, Sparkles, BookOpen } from 'lucide-react';
import EditorToolbar from './EditorToolbar';

export type SelectionAction = 'review' | 'improve' | 'find-citation';

interface EditorProps {
  content: JSONContent | null;
  onUpdate: (content: JSONContent) => void;
  editable?: boolean;
  onSelectionAction?: (action: SelectionAction, selectedText: string) => void;
}

export default function Editor({
  content,
  onUpdate,
  editable = true,
  onSelectionAction,
}: EditorProps) {
  const [menuPos, setMenuPos] = useState<{ x: number; y: number } | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

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
    ],
    editable,
    content: content ?? undefined,
    onUpdate: ({ editor: ed }) => {
      onUpdate(ed.getJSON());
    },
    editorProps: {
      attributes: { class: 'tiptap' },
    },
    onSelectionUpdate: ({ editor: ed }) => {
      const { from, to } = ed.state.selection;
      const text = ed.state.doc.textBetween(from, to, ' ');
      if (text.trim().length > 2 && ed.isEditable) {
        // Get coordinates of selection
        const coords = ed.view.coordsAtPos(from);
        setMenuPos({ x: coords.left, y: coords.top - 45 });
      } else {
        setMenuPos(null);
      }
    },
  });

  // Sync editable
  useEffect(() => {
    if (editor && editor.isEditable !== editable) {
      editor.setEditable(editable);
    }
  }, [editor, editable]);

  // Sync content from props
  useEffect(() => {
    if (!editor || !content) return;
    const currentJSON = JSON.stringify(editor.getJSON());
    const incomingJSON = JSON.stringify(content);
    if (currentJSON !== incomingJSON) {
      editor.commands.setContent(content);
    }
  }, [editor, content]);

  // Close menu on click outside
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuPos(null);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const handleSelectionAction = useCallback(
    (action: SelectionAction) => {
      if (!editor || !onSelectionAction) return;
      const { from, to } = editor.state.selection;
      const selectedText = editor.state.doc.textBetween(from, to, ' ');
      if (selectedText.trim()) {
        onSelectionAction(action, selectedText);
        setMenuPos(null);
      }
    },
    [editor, onSelectionAction],
  );

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
          <EditorContent editor={editor} />
        </div>

        {/* Floating selection menu */}
        {menuPos && (
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
      </div>
    </div>
  );
}

/* ---------- Internal sub-component ---------- */

interface BubbleButtonProps {
  onClick: () => void;
  label: string;
  icon: React.ReactNode;
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
  );
}
