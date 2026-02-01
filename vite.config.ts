import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      ignored: [
        '**/*.db',
        '**/*.db-journal',
        '**/*.db-wal',
        '**/models/**',
        '**/*.log',
        '**/__pycache__/**',
        '**/*.pyc',
      ],
    },
  },
});
