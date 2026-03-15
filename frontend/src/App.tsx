import { useEffect } from 'react'
import { useAppStore } from './stores/appStore'
import { useDocumentStore } from './stores/documentStore'
import { api } from './lib/api'
import { useKeyboard } from './hooks/useKeyboard'
import { Sidebar } from './components/layout/Sidebar'
import { EditorPane } from './components/layout/EditorPane'
import { RightPanel } from './components/layout/RightPanel'
import { ToastContainer } from './components/modals/Toast'

export default function App() {
  const sidebarOpen = useAppStore((s) => s.sidebarOpen)
  const rightPanelOpen = useAppStore((s) => s.rightPanelOpen)
  const setSessions = useAppStore((s) => s.setSessions)
  const setModelInfo = useAppStore((s) => s.setModelInfo)
  const setFiles = useDocumentStore((s) => s.setFiles)
  const setSources = useDocumentStore((s) => s.setSources)
  const setStructure = useDocumentStore((s) => s.setStructure)

  // Global keyboard shortcuts
  useKeyboard()

  useEffect(() => {
    async function bootstrap() {
      const [status, docs, sessions, structure, sources] = await Promise.allSettled([
        api<{ models?: Record<string, string> }>('/api/status'),
        api<any[]>('/api/documents'),
        api<{ sessions: any[] }>('/api/sessions'),
        api<any>('/api/structure'),
        api<any[]>('/api/sources/detailed'),
      ])

      if (status.status === 'fulfilled' && status.value.models) {
        setModelInfo(status.value.models)
      }
      if (docs.status === 'fulfilled') {
        setFiles(Array.isArray(docs.value) ? docs.value : [])
      }
      if (sessions.status === 'fulfilled') {
        setSessions((sessions.value as any).sessions ?? [])
      }
      if (structure.status === 'fulfilled') {
        setStructure((structure.value as any) ?? null)
      }
      if (sources.status === 'fulfilled') {
        setSources(Array.isArray(sources.value) ? sources.value : [])
      }
    }

    bootstrap()
  }, [setSessions, setModelInfo, setFiles, setSources, setStructure])

  return (
    <div className="flex h-full w-full overflow-hidden bg-bg">
      {/* Left sidebar */}
      <div
        className="shrink-0 overflow-hidden border-r border-border transition-[width] duration-200 ease-out"
        style={{ width: sidebarOpen ? 240 : 0 }}
      >
        <div className="flex h-full w-[240px] flex-col">
          <Sidebar />
        </div>
      </div>

      {/* Editor pane -- takes remaining space */}
      <div className="relative flex min-w-0 flex-1 flex-col">
        <EditorPane />
      </div>

      {/* Right panel */}
      <div
        className="shrink-0 overflow-hidden border-l border-border transition-[width] duration-200 ease-out"
        style={{ width: rightPanelOpen ? 280 : 0 }}
      >
        <div className="flex h-full w-[280px] flex-col">
          <RightPanel />
        </div>
      </div>

      <ToastContainer />
    </div>
  )
}
