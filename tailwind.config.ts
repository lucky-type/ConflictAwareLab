import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        panel: {
          DEFAULT: '#0f172a',
          muted: '#111827'
        }
      }
    }
  },
  plugins: [],
} satisfies Config;
