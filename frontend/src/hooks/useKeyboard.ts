/**
 * Global keyboard shortcuts hook.
 *
 * Ctrl+1-4 → mode switch (chat/draft/review/synthesize)
 * Ctrl+5-8 → mode switch (cite/transition/consistency/structure)
 * Ctrl+D   → toggle right panel
 * Ctrl+B   → toggle sidebar
 * Escape   → abort streaming
 */
import { useEffect } from 'react'
import { useAppStore } from '../stores/appStore'
import type { ModeId } from '../lib/constants'

const MODE_KEYS: Record<string, ModeId> = {
  '1': 'chat',
  '2': 'draft',
  '3': 'review',
  '4': 'synthesize',
  '5': 'cite',
  '6': 'transition',
  '7': 'consistency',
  '8': 'structure',
}

export function useKeyboard() {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const ctrl = e.ctrlKey || e.metaKey

      // Mode switching: Ctrl+1 through Ctrl+8
      if (ctrl && MODE_KEYS[e.key]) {
        e.preventDefault()
        useAppStore.getState().setMode(MODE_KEYS[e.key])
        return
      }

      // Toggle right panel: Ctrl+D
      if (ctrl && e.key === 'd') {
        e.preventDefault()
        useAppStore.getState().toggleRightPanel()
        return
      }

      // Toggle sidebar: Ctrl+Shift+B
      if (ctrl && e.shiftKey && e.key.toLowerCase() === 'b') {
        e.preventDefault()
        useAppStore.getState().toggleSidebar()
        return
      }

      // Abort streaming: Escape
      if (e.key === 'Escape') {
        const ctrl = useAppStore.getState().abortController
        if (ctrl) {
          ctrl.abort()
        }
        return
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])
}
