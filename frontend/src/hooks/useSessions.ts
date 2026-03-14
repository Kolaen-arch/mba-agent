import { useCallback } from 'react'
import { api } from '../lib/api'
import { useAppStore } from '../stores/appStore'

export function useSessions() {
  const { setSessions, setSession, setMessages, setMode } = useAppStore()

  const loadSessions = useCallback(async () => {
    try {
      const data = await api('/api/sessions')
      setSessions(data.sessions || [])
    } catch (e) {
      console.error('Failed to load sessions:', e)
    }
  }, [])

  const loadSession = useCallback(async (sessionId: string) => {
    try {
      const data = await api(`/api/sessions/${sessionId}`)
      setSession(data.session)
      setMessages(data.messages || [])
      if (data.session?.mode) setMode(data.session.mode)
    } catch (e) {
      console.error('Failed to load session:', e)
    }
  }, [])

  const newSession = useCallback(() => {
    setSession(null)
    setMessages([])
    setMode('chat')
  }, [])

  const deleteSession = useCallback(async (sessionId: string) => {
    try {
      await api(`/api/sessions/${sessionId}`, { method: 'DELETE' })
      await loadSessions()
      const current = useAppStore.getState().session
      if (current?.id === sessionId) {
        newSession()
      }
    } catch (e) {
      console.error('Failed to delete session:', e)
    }
  }, [])

  return { loadSessions, loadSession, newSession, deleteSession }
}
