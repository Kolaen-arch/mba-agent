import { useEffect, useState, useCallback } from 'react'

interface ToastMessage {
  id: number
  text: string
}

let toastId = 0
let addToastFn: ((text: string) => void) | null = null

export function toast(text: string) {
  addToastFn?.(text)
}

export function ToastContainer() {
  const [toasts, setToasts] = useState<ToastMessage[]>([])

  const addToast = useCallback((text: string) => {
    const id = ++toastId
    setToasts((prev) => [...prev, { id, text }])
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, 3000)
  }, [])

  useEffect(() => {
    addToastFn = addToast
    return () => { addToastFn = null }
  }, [addToast])

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex flex-col gap-2 items-center">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="px-4 py-2 rounded-lg shadow-lg text-sm font-medium animate-in fade-in slide-in-from-bottom-2"
          style={{
            background: 'var(--bg-dark, #2D2A26)',
            color: 'var(--text-inverse, #FDFCFA)',
            fontFamily: 'var(--font-ui)',
          }}
        >
          {t.text}
        </div>
      ))}
    </div>
  )
}
