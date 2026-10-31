import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built into web/dist, which Flask serves. During `npm run dev` the API calls
// are proxied to Flask so the UI can hot-reload against real data.
export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:5000' },
  },
})
