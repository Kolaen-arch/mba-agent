import { create } from 'zustand'
import type { JSONContent } from '@tiptap/react'

export interface DocFile {
  path: string
  filename: string
  relative_path: string
  word_count: number
  page_estimate: number
}

export interface SourceFile {
  file: string
  label: string
  chunks: number
}

export interface DocData {
  full_text: string
  metadata: { word_count: number }
  sections: { heading: string; paragraphs: string[] }[]
}

export interface PaperSection {
  id: string
  title: string
  parent_id: string | null
  status: string
  word_count: number
  target_words: number
  summary: string
  order: number
}

export interface PaperStructure {
  title: string
  research_question: string
  red_thread: string
  sections: PaperSection[]
  progress: {
    total_words: number
    target_words: number
    pct: number
    sections_complete: number
    sections_total: number
  }
}

interface DocumentState {
  // Document
  files: DocFile[]
  sources: SourceFile[]
  currentPath: string
  currentDoc: DocData | null
  editorContent: JSONContent | null

  // Diff / draft
  pendingDraft: string
  diffMode: boolean

  // Structure
  structure: PaperStructure | null

  // Actions
  setFiles: (f: DocFile[]) => void
  setSources: (s: SourceFile[]) => void
  setCurrentPath: (p: string) => void
  setCurrentDoc: (d: DocData | null) => void
  setEditorContent: (c: JSONContent | null) => void
  setPendingDraft: (d: string) => void
  setDiffMode: (v: boolean) => void
  setStructure: (s: PaperStructure | null) => void
}

export const useDocumentStore = create<DocumentState>((set) => ({
  files: [],
  sources: [],
  currentPath: '',
  currentDoc: null,
  editorContent: null,
  pendingDraft: '',
  diffMode: false,
  structure: null,

  setFiles: (files) => set({ files }),
  setSources: (sources) => set({ sources }),
  setCurrentPath: (currentPath) => set({ currentPath }),
  setCurrentDoc: (currentDoc) => set({ currentDoc }),
  setEditorContent: (editorContent) => set({ editorContent }),
  setPendingDraft: (pendingDraft) => set({ pendingDraft }),
  setDiffMode: (diffMode) => set({ diffMode }),
  setStructure: (structure) => set({ structure }),
}))
