import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import { execSync } from 'child_process'

// Resolve the main git repo root so builds always land where Flask serves from,
// even when running from a worktree.
function getRepoRoot(): string {
  try {
    // --git-common-dir returns the shared .git dir (e.g. "C:/Git/repo/.git")
    // Its parent is always the main repo root, whether in a worktree or not.
    const commonDir = execSync('git rev-parse --git-common-dir', {
      encoding: 'utf-8',
      cwd: __dirname,
    }).trim()
    return path.resolve(commonDir, '..')
  } catch {
    return path.resolve(__dirname, '..')
  }
}

const repoRoot = getRepoRoot()

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: path.join(repoRoot, 'mba_agent', 'web', 'static', 'dist'),
    emptyOutDir: true,
  },
})
