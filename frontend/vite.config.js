import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,           // ← hace que Vite escuche en 0.0.0.0
    port: 5173,           // ← puerto explícito
    strictPort: true      // ← falla si el puerto ya está en uso
  }
})