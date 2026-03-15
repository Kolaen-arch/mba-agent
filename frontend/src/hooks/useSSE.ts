import { useCallback } from 'react'
import { useAppStore } from '../stores/appStore'
import { apiRaw } from '../lib/api'

interface SSECallbacks {
  onModel?: (model: string, thinkingEnabled: boolean) => void
  onThinkingStart?: () => void
  onThinking?: (text: string) => void
  onThinkingDone?: () => void
  onText?: (text: string) => void
  onDone?: (data: any) => void
  onError?: (error: string) => void
}

export function useSSE() {
  const {
    session, mode, sectionId, modelOverride, thinkOverride,
    setLoading, setStreamText, appendStreamText,
    setThinkText, appendThinkText, setLastResp,
    setAbortController, setSession, addMessage,
  } = useAppStore()

  const send = useCallback(async (message: string, callbacks?: SSECallbacks) => {
    if (!message.trim()) return

    const abortCtrl = new AbortController()
    setAbortController(abortCtrl)
    setLoading(true)
    setStreamText('')
    setThinkText('')

    // Add user message
    addMessage({ role: 'user', content: message })

    try {
      const payload: any = {
        session_id: session?.id || null,
        message,
        mode,
        section_id: sectionId || undefined,
      }
      if (modelOverride) payload.model_override = modelOverride
      if (thinkOverride) payload.thinking_override = parseInt(thinkOverride)

      const res = await apiRaw('/api/chat/stream', {
        method: 'POST',
        body: JSON.stringify(payload),
        signal: abortCtrl.signal,
      })

      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let evtType = ''
      let accumulated = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            evtType = line.slice(7).trim()
          } else if (line.startsWith('data: ') && evtType) {
            try {
              const data = JSON.parse(line.slice(6))

              switch (evtType) {
                case 'model':
                  callbacks?.onModel?.(data.model || '', data.thinking_enabled)
                  break
                case 'thinking_start':
                  callbacks?.onThinkingStart?.()
                  break
                case 'thinking':
                  appendThinkText(data.text)
                  callbacks?.onThinking?.(data.text)
                  break
                case 'thinking_done':
                  callbacks?.onThinkingDone?.()
                  break
                case 'text':
                  accumulated += data.text
                  appendStreamText(data.text)
                  callbacks?.onText?.(data.text)
                  break
                case 'done':
                  // Save last response
                  setLastResp(accumulated)
                  // Create session if new
                  if (!session && data.session_id) {
                    setSession({ id: data.session_id, title: message.slice(0, 50), mode, created_at: new Date().toISOString() })
                  }
                  // Add assistant message
                  addMessage({ role: 'assistant', content: accumulated, model: data.model })
                  callbacks?.onDone?.(data)
                  break
                case 'error':
                  callbacks?.onError?.(data.error)
                  break
              }
            } catch (e) {
              // Skip malformed JSON
            }
            evtType = ''
          }
        }
      }
    } catch (e: any) {
      if (e.name === 'AbortError') {
        // User cancelled
      } else {
        callbacks?.onError?.(e.message)
      }
    } finally {
      setAbortController(null)
      setLoading(false)
    }
  }, [session, mode, sectionId, modelOverride, thinkOverride])

  const abort = useCallback(() => {
    const ctrl = useAppStore.getState().abortController
    if (ctrl) ctrl.abort()
  }, [])

  return { send, abort }
}
