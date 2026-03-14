import { create } from 'zustand'
import type { ModeId } from '../lib/constants'

export interface Session {
  id: string
  title: string
  mode: string
  created_at: string
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  context_sources?: string[]
  model?: string
  timestamp?: string
}

interface AppState {
  // Session
  session: Session | null
  sessions: Session[]
  messages: Message[]

  // Mode & overrides
  mode: ModeId
  modelOverride: string
  thinkOverride: string
  sectionId: string

  // UI state
  loading: boolean
  sidebarOpen: boolean
  rightPanelOpen: boolean
  streamText: string
  thinkText: string
  lastResp: string
  abortController: AbortController | null

  // Model info from status
  modelInfo: Record<string, string> | null

  // Actions
  setMode: (mode: ModeId) => void
  setSession: (s: Session | null) => void
  setSessions: (s: Session[]) => void
  setMessages: (m: Message[]) => void
  addMessage: (m: Message) => void
  setModelOverride: (v: string) => void
  setThinkOverride: (v: string) => void
  setSectionId: (id: string) => void
  setLoading: (v: boolean) => void
  toggleSidebar: () => void
  toggleRightPanel: () => void
  setStreamText: (v: string) => void
  appendStreamText: (v: string) => void
  setThinkText: (v: string) => void
  appendThinkText: (v: string) => void
  setLastResp: (v: string) => void
  setAbortController: (c: AbortController | null) => void
  setModelInfo: (info: Record<string, string>) => void
  reset: () => void
}

const initialState = {
  session: null,
  sessions: [],
  messages: [],
  mode: 'chat' as ModeId,
  modelOverride: '',
  thinkOverride: '',
  sectionId: '',
  loading: false,
  sidebarOpen: true,
  rightPanelOpen: true,
  streamText: '',
  thinkText: '',
  lastResp: '',
  abortController: null,
  modelInfo: null,
}

export const useAppStore = create<AppState>((set) => ({
  ...initialState,

  setMode: (mode) => set({ mode }),
  setSession: (session) => set({ session }),
  setSessions: (sessions) => set({ sessions }),
  setMessages: (messages) => set({ messages }),
  addMessage: (m) => set((s) => ({ messages: [...s.messages, m] })),
  setModelOverride: (modelOverride) => set({ modelOverride }),
  setThinkOverride: (thinkOverride) => set({ thinkOverride }),
  setSectionId: (sectionId) => set({ sectionId }),
  setLoading: (loading) => set({ loading }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  toggleRightPanel: () => set((s) => ({ rightPanelOpen: !s.rightPanelOpen })),
  setStreamText: (streamText) => set({ streamText }),
  appendStreamText: (v) => set((s) => ({ streamText: s.streamText + v })),
  setThinkText: (thinkText) => set({ thinkText }),
  appendThinkText: (v) => set((s) => ({ thinkText: s.thinkText + v })),
  setLastResp: (lastResp) => set({ lastResp }),
  setAbortController: (abortController) => set({ abortController }),
  setModelInfo: (modelInfo) => set({ modelInfo }),
  reset: () => set({ ...initialState, sidebarOpen: true, rightPanelOpen: true }),
}))
